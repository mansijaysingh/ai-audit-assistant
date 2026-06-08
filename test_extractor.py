from utils.pdf_reader import extract_document
from utils.rag_builder import build_vector_store
from agents.extractor import retrieve_financial_item, extract_financial_data

texts_dict={
  "current": extract_document("BSnew.pdf")
}


retriever=build_vector_store(texts_dict)

context= retrieve_financial_item(
  "net profit gross profit turnover firm name",
  retriever
)

result=extract_financial_data(
  context
)


print(result)