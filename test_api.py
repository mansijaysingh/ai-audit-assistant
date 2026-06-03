from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings=OpenAIEmbeddings(
  model="text-embedding-3-small"
)

result=embeddings.embed_query("Hello AI Financial Audit Assistant")

print(f"Vector Length: {len(result)}")
print(result[:5])