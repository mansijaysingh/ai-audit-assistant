from utils.pdf_reader import extract_document

pdf_path = "balance_Sheet.pdf"

document_text = extract_document(pdf_path)

print(document_text[:3000])