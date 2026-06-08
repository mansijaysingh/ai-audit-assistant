from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI(
  model="gpt-4o-mini"
)


def retrieve_financial_item(
    item_name,
    retriever
):
  
   docs=retriever.invoke(item_name)

   context= "\n\n".join(
      doc.page_content
      for doc in docs
   )

   return context