# file_context_0: autocrew.py

# Standard library imports
import argparse
import csv
import io
import json
import logging
import os
import requests
from datetime import datetime
from typing import Any, Dict, List

# Related third-party imports
from packaging import version
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools import tool
from langchain_community.llms import Ollama
from langchain_core.callbacks import base
from unstructured.partition.html import partition_html

# Local application/library-specific imports
from crewai import Agent, Crew, Process, Task

# Including this due to the hacks to follow:
ollama_openhermes = Ollama(model="openhermes",
                           base_url=os.environ['OLLAMA_HOST'])

# Autocrew version
autocrew_version = "1.3.1"

ollama_host = os.getenv('OLLAMA_HOST')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_ollama(model='openhermes', use_ollama_host=True):
   if use_ollama_host:
      ollama_base_url = os.getenv('OLLAMA_HOST')
      if not ollama_base_url:
         raise EnvironmentError("OLLAMA_HOST environment variable not set")
      return Ollama(base_url=ollama_base_url,
                    model=model,
                    verbose=True,
                    callback_manager=CallbackManager(
                        [StreamingStdOutCallbackHandler()]))
   else:
      ollama_base_url = os.getenv('OLLAMA_HOST')
      return Ollama(base_url=ollama_base_url,
                    model=model,
                    verbose=True,
                    callback_manager=CallbackManager(
                        [StreamingStdOutCallbackHandler()]))


# Call the initialize_ollama function
ollama = initialize_ollama()


def get_agent_data(ollama, overall_goal, delimiter):
   instruction = (
       f'Create a dataset in a CSV format with each field enclosed in double quotes, for a team of agents with the goal: "{overall_goal}". '
       f'Use the delimiter "{delimiter}" to separate the fields. '
       'Include columns "role", "goal", "backstory", "assigned_task", "allow_delegation". '
       'Each agent\'s details should be in quotes to avoid confusion with the delimiter. '
       'Provide a single-word role, individual goal, brief backstory, assigned task, and delegation ability (True/False) for each agent.'
   )
   response = ollama.invoke(
       instruction.format(overall_goal=overall_goal, delimiter=delimiter))
   return response


def get_next_crew_name(overall_goal, greek_alphabets):
   # Replace spaces with hyphens in overall_goal for filename matching
   formatted_goal = overall_goal.replace(" ", "-")

   # Get all CSV files that include the overall_goal in their name
   existing_csv_files = [
       f for f in os.listdir(os.getcwd())
       if f.endswith('.csv') and formatted_goal in f
   ]

   # Find the highest Greek alphabet index used in these filenames
   existing_indices = []
   for file_name in existing_csv_files:
      for greek_alpha in greek_alphabets:
         if greek_alpha in file_name:
            existing_indices.append(greek_alphabets.index(greek_alpha))
            break  # Stop after finding the first matching Greek alphabet

   # Determine the next Greek alphabet index
   next_index = max(existing_indices) + 1 if existing_indices else 0
   return greek_alphabets[next_index % len(greek_alphabets)]


def save_csv_output(response, overall_goal, greek_alphabets):
   timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
   crew_name = get_next_crew_name(overall_goal, greek_alphabets)
   clean_crew_name = crew_name.strip('"')  # Remove quotes for file name
   file_name = f'crewai-autocrew-{timestamp}-{overall_goal[:40].replace(" ", "-")}-{clean_crew_name}.csv'
   file_path = os.path.join(os.getcwd(), file_name)

   # Split the response into lines
   lines = response.split('\n')

   # Write the modified response to the file
   with open(file_path, 'w') as file:
      # Write the header row
      file.write("crew_name," + lines[0] + '\n')

      # Modify and write the data rows
      for line in lines[1:]:
         if line.strip():
            # Add crew_name to each line to include it in the concatenated CSV
            modified_line = f'"{crew_name}",{line}\n'
            file.write(modified_line)

   return file_path


def parse_csv_data(response, delimiter=',', filename=''):
   header = ['role', 'goal', 'backstory', 'assigned_task', 'allow_delegation']
   agents_data = []

   csv_data = csv.reader(io.StringIO(response), delimiter=delimiter)
   lines = list(csv_data)

   header_line = lines[0]
   header_mapping = {h.lower(): h for h in header}
   header_indices = [header_mapping.get(h.lower()) for h in header_line]

   if not header_indices:
      raise ValueError('Header component missing in CSV data')

   for line in lines[1:]:
      agent_data = {}
      for i, value in enumerate(line):
         header_name = header_indices[i]
         if header_name:
            if header_name == 'assigned_task':
               # Replace commas with 'and' within square brackets and remove the square brackets
               value = value.replace(',',
                                     ' and ').replace('[',
                                                      '').replace(']', '')
               agent_data[header_name] = value.strip()
            else:
               agent_data[header_name] = value.strip('"')
      if 'role' not in agent_data or not agent_data['role']:
         raise ValueError('Role component missing in CSV data')
      agent_data['filename'] = filename  # Add the filename to the agent data
      agents_data.append(agent_data)

   return agents_data


def define_agent(agent, search_tool):
   role_var = agent['role'].replace(' ', '_').replace('-', '_').replace(
       '.', '_').replace(' ', '')
   role_value = agent['role'].replace('"', '\\"').replace("'", "\\'")
   backstory = agent['backstory'].replace('"', '\\"').replace("'", "\\'")
   delegation = 'True' if agent['allow_delegation'] == 'True' else 'False'
   return (f'{role_var} = Agent(\n'
           f'    role="{role_value}",\n'
           f'    goal="{agent["goal"]}",\n'
           f'    backstory="{backstory}",\n'
           f'    verbose=True,\n'
           f'    allow_delegation={delegation},\n'
           f'    llm=ollama_openhermes,\n'
           f'    tools={search_tool}\n'
           ')\n\n')


def get_task_var_name(role):
   return f'task_{role.replace(" ", "_").replace("-", "_").replace(".", "_")}'


def define_task(agent):
   task_var = get_task_var_name(agent['role'])

   # Escape double quotes in assigned_task if needed
   task_description = agent["assigned_task"].strip().replace('"', '\\"')

   return (
       f'{task_var} = Task(\n'
       f' description="{task_description}",\n'
       f' agent={agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")},\n'
       ' verbose=True,\n'
       ')\n\n')


def generate_crew_tasks(agents_data):
   return ', '.join([
       f'task_{agent["role"].replace(" ", "_").replace("-", "_").replace(".", "_")}'
       for agent in agents_data
   ])


def write_crewai_script(agents_data, crew_tasks, file_name, use_ollama_host):
   crew_agents = ', '.join([
       agent['role'].replace(' ', '_').replace('-', '_').replace('.', '_')
       for agent in agents_data
   ])

   with open(file_name, 'w') as file:
      file.write(
          'import os\n'
          'from langchain_community.chat_models import ChatOpenAI\n'
          'from langchain_community.llms import Ollama\n'
          'from browser_tools import BrowserTools\n'
          'from search_tools import SearchTools\n'
          'from crewai import Agent, Task, Crew, Process\n\n'
          'os.environ["OPENAI_API_KEY"] = "your_OPENAI_api_key_here"\n\n')

      if use_ollama_host:
         file.write(f'ollama_host = "{ollama_host}"\n'
                    )  # Write the ollama_host variable to the generated script
         file.write(
             'ollama_openhermes = Ollama(model="openhermes", base_url=ollama_host)\n'
         )  # Use ollama_host to initialize Ollama
      else:
         file.write('ollama_openhermes = Ollama(model="openhermes")\n')

      file.write(
          'search_tool = [BrowserTools.scrape_and_summarize_website, SearchTools.search_internet]\n\n'
      )

      for agent in agents_data:
         file.write(define_agent(agent, "search_tool"))
         file.write('\n')

      for agent in agents_data:
         file.write(define_task(agent))
         file.write('\n')

      file.write('crew = Crew(\n'
                 f'    agents=[{crew_agents}],\n'
                 f'    tasks=[{crew_tasks}],\n'
                 '    verbose=True,\n'
                 '    process=Process.sequential,\n'
                 ')\n\n'
                 '# Kickoff the crew tasks\n'
                 'result = crew.kickoff()\n\n'
                 '# Handle the "result" as needed\n')


def check_latest_version():
   try:
      response = requests.get(
          'https://raw.githubusercontent.com/yanniedog/autocrew/main/autocrew.py'
      )
      response.raise_for_status()
      script_content = response.text
      version_line = next(line for line in script_content.split('\n')
                          if line.startswith('autocrew_version = '))
      latest_version = version_line.split('=')[1].strip().strip('"')

      if version.parse(latest_version) > version.parse(autocrew_version):
         return latest_version
      else:
         return None

   except Exception as e:
      print(f'Error checking the latest version: {e}')
      return None


def rank_crews(ollama, csv_file_paths, overall_goal, verbose=False):
   ranked_crews = []
   overall_summary = ''
   concatenated_csv_data = 'crew_name,role,goal,backstory,assigned_task,allow_delegation\n'

   for file_path in csv_file_paths:
      try:
         with open(file_path, 'r') as file:
            csv_data = file.read().strip()

         if csv_data.count('\n') < 1:
            continue

         concatenated_csv_data += csv_data[csv_data.index('\n') + 1:] + '\n'

      except Exception as e:
         print(f"Error processing file {file_path}: {e}")

   if concatenated_csv_data.strip(
   ) == 'crew_name,role,goal,backstory,assigned_task,allow_delegation':
      print("Warning: No valid data found in the provided CSV files.")
      return [], "No ranking could be performed due to insufficient data."

   # Convert the concatenated CSV data to a JSON object
   json_data = []
   csv_reader = csv.DictReader(io.StringIO(concatenated_csv_data))
   for row in csv_reader:
      json_data.append(row)
   json_data_str = json.dumps(json_data)

   if verbose:
      print('\nConcatenated CSV Data:')
      print(concatenated_csv_data)

   crew_names_str = ', '.join([
       os.path.basename(file_path).split('-')[-1].split('.')[0]
       for file_path in csv_file_paths
   ])

   # Construct and print the Ollama prompt with the crew names
   prompt = (
       f"Analyze the following list of crews ({crew_names_str}) to determine their suitability for successfully completing the task: "
       f"{overall_goal}. The crews are represented in a JSON object format: {json_data_str}. "
       "Please provide a ranking of the crews by their names, with the most suitable crew listed first. "
       "Also, provide a brief critique for each crew, highlighting their strengths and weaknesses."
   )

   if verbose:
      print("Prompt to be sent to Ollama:\n", prompt)

   # Invoke Ollama with the prompt and JSON object
   ranked_crew = ollama.invoke(prompt)

   ranked_crews.append((concatenated_csv_data, ranked_crew))
   overall_summary += f'\n\nCrews in the following CSV files:\n'
   for file_path in csv_file_paths:
      overall_summary += f'{file_path}\n'
   overall_summary += f'Ranking: {ranked_crew}\n'

   overall_summary += f'\nOverall Summary:\n'
   overall_summary += f'Ollama has ranked the crews based on their likelihood of success.\n'
   overall_summary += f'It has provided a critique for each crew, highlighting their strengths and weaknesses.\n'
   overall_summary += f'The ranking and critique can be used to make informed decisions about the crews.\n'

   return ranked_crews, overall_summary


def main():
   greek_alphabets = [
       "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
       "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
       "sigma", "tau", "upsilon"
   ]
   print(f"\nAutocrew (v{autocrew_version}) for CrewAI\n")

   print(
       "\nTo see the available command line parameters, type: python3 autocrew.py --help\n"
   )

   parser = argparse.ArgumentParser(description='CrewAI Autocrew Script')
   parser.add_argument('overall_goal',
                       nargs='?',
                       type=str,
                       help='The overall goal for the crew')
   parser.add_argument('-a',
                       '--auto_run',
                       action='store_true',
                       help='Automatically run the generated script')
   parser.add_argument(
       '-m',
       '--multiple',
       type=int,
       metavar='NUM',
       help=
       'Create NUM number of CrewAI scripts for the same overall goal. Example: -m 3'
   )
   parser.add_argument(
       '-r',
       '--ranking',
       action='store_true',
       help=
       'Perform ranking only based on existing CSV files --> currently EXPERIMENTAL'
   )
   parser.add_argument('-v',
                       '--verbose',
                       action='store_true',
                       help='Enable verbose output')
   parser.add_argument(
       '--use_ollama_host',
       action='store_true',
       help='Use OLLAMA_HOST from the original script in the generated script')

   args = parser.parse_args()

   # Initial summary of actions based on arguments
   print("\nInitial Summary of Actions:")
   if args.overall_goal:
      print(f"  - Overall goal specified: {args.overall_goal}")
   if args.auto_run:
      print("  - The script(s) will be automatically run after generation.")
   if args.multiple:
      print(f"  - Number of scripts to generate: {args.multiple}")
   if args.ranking:
      print(
          "  - Ranking mode activated. Existing CSV files will be used for ranking."
      )
   if args.verbose:
      print(
          "  - Verbose mode activated. Additional details will be provided during execution."
      )
   if args.use_ollama_host:
      print(
          "  - Use OLLAMA_HOST from the original script in the generated script."
      )
   print()

   if args.ranking and not args.overall_goal:
      print(
          "Warning: Ranking mode requires an overall goal. Please provide an overall goal using the command line or by entering it when prompted."
      )

   if args.overall_goal is None:
      overall_goal = input('\nPlease specify the overall goal: \n')
   else:
      overall_goal = args.overall_goal

   if args.multiple:
      num_scripts = args.multiple
   else:
      num_scripts = 1

   csv_file_paths = []  # Initialize the list of CSV file paths

   ollama = initialize_ollama(use_ollama_host=args.use_ollama_host)

   # Generate the specified number of scripts using the -m option
   if not args.ranking or args.multiple:
      existing_csv_files = [
          f for f in os.listdir(os.getcwd())
          if f.endswith('.csv') and overall_goal in f and any(
              greek_alpha in f for greek_alpha in greek_alphabets)
      ]
      existing_indices = [
          greek_alphabets.index(greek_alpha) for f in existing_csv_files
          for greek_alpha in greek_alphabets if greek_alpha in f
      ]
      starting_index = max(existing_indices) + 1 if existing_indices else 0

      for i in range(starting_index, starting_index + num_scripts):
         print(
             f"\nStarting script generation {i + 1} of {num_scripts} for the goal: '{overall_goal}'\n"
         )

         if args.verbose:
            print("\nSending request to Ollama for agent data...\n")
         response = get_agent_data(ollama, overall_goal, delimiter=',')
         if not response:
            raise ValueError('No response from Ollama')

         file_path = save_csv_output(response, overall_goal, greek_alphabets)
         csv_file_paths.append(file_path)  # Store the CSV file path
         agents_data = parse_csv_data(response,
                                      delimiter=',',
                                      filename=file_path)
         if not agents_data:
            raise ValueError('No agent data parsed')

         file_name = os.path.basename(file_path).replace('.csv', '.py')
         crewai_script_path = os.path.join(os.getcwd(), file_name)
         crew_tasks = generate_crew_tasks(agents_data)

         write_crewai_script(agents_data, crew_tasks, crewai_script_path,
                             args.use_ollama_host)
         print(f"\nScript {i + 1} written to {crewai_script_path}\n")

         if args.auto_run:
            print(f'\nAutomatically running script {i + 1}...\n')
            os.system(f'python3 {crewai_script_path}')

   # Perform ranking using the -r option
   if args.ranking:
      print("Sending ranking request to Ollama...\n")
      # Use the CSV files generated above if multiple scripts were generated
      if not csv_file_paths and args.overall_goal:
         # If no new CSV files were generated, look for existing ones
         csv_file_paths = [
             f for f in os.listdir(os.getcwd())
             if f.endswith('.csv') and args.overall_goal.replace(" ", "-") in f
             and any(greek_alpha in f for greek_alpha in greek_alphabets)
         ]

      if csv_file_paths:
         ranked_crews, overall_summary = rank_crews(ollama, csv_file_paths,
                                                    overall_goal, args.verbose)
         print(overall_summary)

         import re
         top_crew_name_search = re.search(r'"(.+?)"', overall_summary)
         if top_crew_name_search:
            top_crew_name = top_crew_name_search.group(1)
         else:
            print(
                "Error: Top-ranked crew name not found in the overall summary."
            )

         if args.auto_run:
            overall_goal_formatted = overall_goal.replace(" ", "-")
            script_files = [
                f for f in os.listdir(os.getcwd()) if f.endswith('.py')
            ]

            for script_file in script_files:
               if overall_goal_formatted in script_file and top_crew_name in script_file:
                  top_script_path = os.path.join(os.getcwd(), script_file)
                  break

         # Execute the top-ranked script
         if args.verbose:
            print(f"\nTop-ranked crew name: {top_crew_name}")
            print("\nAvailable script files in the directory:")
            script_files = [
                f for f in os.listdir(os.getcwd()) if f.endswith('.py')
            ]
            for script_file in script_files:
               print(f"  - {script_file}")
            print(
                f"\nSearching for script file with overall_goal_formatted: {overall_goal_formatted} and top_crew_name: {top_crew_name}"
            )

         if top_script_path:
            print(
                f'\nAutomatically running the top-ranked script: {top_script_path}\n'
            )
            os.system(f'python3 {top_script_path}')
         else:
            print(
                "\nTop-ranked script not found. Please ensure the script files are in the correct directory.\n"
            )


if __name__ == '__main__':
   main()
