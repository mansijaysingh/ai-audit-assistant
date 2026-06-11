from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

llm=ChatOpenAI(
  model="gpt-4o-mini"
)


FIELDS = [
    "firm_name",
    "ca_name",
    "pan_number",
    "net_profit",
    "gross_profit",
    "turnover"
]


