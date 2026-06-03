from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

llm=OpenAI(model="gpt-4o-mini")

response= llm.invoke("Say hello")

print(response)
