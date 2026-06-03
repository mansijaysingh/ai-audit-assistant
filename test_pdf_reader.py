from utils.pdf_reader import extract_text, extract_table

pdf_path= "sample.pdf"

text=extract_text(pdf_path)

print(text[:2000])


tables= extract_table(pdf_path)

print("\nTABLES FOUND:")
print(len(tables))

if tables:
  print(tables[0])