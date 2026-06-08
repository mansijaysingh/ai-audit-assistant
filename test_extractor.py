from utils.pdf_reader import extract_document
from utils.rag_builder import build_vectorstore
from agents.extractor import get_context_for_extraction, extract_financial_data


# Change these file names according to your PDFs
current_pdf = "BSnew.pdf"
previous_pdf = "BSold.pdf"
cb3_pdf = "3cb.pdf"


current_text = extract_document(current_pdf)
previous_text = extract_document(previous_pdf)
cb3_text = extract_document(cb3_pdf)


current_retriever = build_vectorstore({
    "current": current_text
})

previous_retriever = build_vectorstore({
    "previous": previous_text
})

cb3_retriever = build_vectorstore({
    "3cb": cb3_text
})


current_context = get_context_for_extraction(
    current_retriever,
    "current"
)

previous_context = get_context_for_extraction(
    previous_retriever,
    "previous"
)

cb3_context = get_context_for_extraction(
    cb3_retriever,
    "3cb"
)


current_data = extract_financial_data(
    current_context,
    "current"
)

previous_data = extract_financial_data(
    previous_context,
    "previous"
)

cb3_data = extract_financial_data(
    cb3_context,
    "3cb"
)


print("\n========== CURRENT DATA ==========\n")
print(current_data)

print("\n========== PREVIOUS DATA ==========\n")
print(previous_data)

print("\n========== 3CB DATA ==========\n")
print(cb3_data)