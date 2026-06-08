from utils.pdf_reader import extract_document
from utils.rag_builder import build_vector_store, get_relevant_chunks


texts_dict= {
  "current": extract_document("BSnew.pdf"),
    "previous": extract_document("BSold.pdf"),
    "3cb": extract_document("3cb.pdf")
}

retriever=build_vector_store(texts_dict)

docs=get_relevant_chunks(
  "what is the closing stock?",
  retriever
)

for i, doc in enumerate(docs, start=1):


    print("\n====================")
    print(f"CHUNK {i}")
    print("SOURCE:", doc.metadata)
    print(doc.page_content[:500])