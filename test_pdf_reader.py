from utils.pdf_reader import extract_text

pdf_path= "sample.pdf"

text=extract_text(pdf_path)

print(text[:2000])