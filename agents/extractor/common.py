from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

llm=ChatOpenAI(
  model="gpt-4o-mini"
)

parser=JsonOutputParser()


def run_json_chain(prompt: PromptTemplate, context: str, field_list: str):

  chain= llm | prompt | parser

  result=chain.invoke(
    {
      "retrieved chunks":context,
      "field_list": field_list
    }
  )

  return result