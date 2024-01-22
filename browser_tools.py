# file: browser_tools.py

# Standard library imports
import os
import json
import requests
from unstructured.partition.html import partition_html
from langchain_community.llms import Ollama
from crewai import Agent, Task
from langchain.tools import tool


# BrowserTools class definition moving to its own file
class BrowserTools():

   @tool("Scrape website content")
   def scrape_and_summarize_website(website):
      """Useful to scrape and summarize a website content"""
      url = os.environ['BROWSERLESS_HOST']
      ollama_openhermes = Ollama(model="openhermes",
                                 base_url=os.environ['OLLAMA_HOST'])
      payload = json.dumps({"url": website})
      headers = {
          'cache-control': 'no-cache',
          'content-type': 'application/json'
      }
      response = requests.request("POST", url, headers=headers, data=payload)
      elements = partition_html(text=response.text)
      content = "\n\n".join([str(el) for el in elements])
      content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
      summaries = []
      for chunk in content:
         agent = Agent(
             role='Principal Researcher',
             goal=
             'Do amazing research and summaries based on the content you are working with',
             backstory=
             "You're a Principal Researcher at a big company and you need to do research about a given topic.",
             allow_delegation=False,
             llm=ollama_openhermes)
         task = Task(
             agent=agent,
             description=
             f'Analyze and summarize the content below, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}'
         )
         summary = task.execute()
         summaries.append(summary)
      return "\n\n".join(summaries)
