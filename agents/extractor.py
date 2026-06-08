from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI(
  model="gpt-4o-mini"
)


prompt=PromptTemplate.from_template(
   """
You are a financial audit data extraction expert.

Extract the following fields from the context.

Fields:

firm_name
ca_name
pan_number
net_profit
gross_profit
turnover

Instructions:

- Return ONLY valid JSON.
- If a field is not found, return null.
- Do not explain anything.
- Do not return markdown.

Context:

{context}
"""
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


def extract_financial_data(context):
   
   parser=JsonOutputParser()

   chain=prompt | llm | parser

   result= chain.invoke(
      {
         "context" : context
      }
   )

   return result