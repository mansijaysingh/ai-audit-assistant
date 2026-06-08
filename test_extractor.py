from utils.pdf_reader import extract_document
from utils.rag_builder import build_vector_store
from agents.extractor import retrieve_financial_item

texts_dict={
  "current": extract_document("BSnew.pdf")
}


retriever=build_vector_store(texts_dict)

result= retrieve_financial_item(
  "net profit",
  retriever
)

print(result[:3000])