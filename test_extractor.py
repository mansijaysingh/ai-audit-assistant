from utils.pdf_reader import extract_document
from utils.rag_builder import build_vector_store
from agents.extractor import retrieve_financial_item, extract_financial_data, get_context_for_extraction

texts_dict={
  "current": extract_document("BSnew.pdf")
}


retriever=build_vector_store(texts_dict)

context= get_context_for_extraction (
  retriever
)

result=extract_financial_data(
  context
)


print(result)