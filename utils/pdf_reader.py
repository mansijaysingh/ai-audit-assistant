import pdfplumber

def extract_text(file_path):
  full_text=""

  with pdfplumber.open(file_path)as pdf:

    for page in pdf.pages:

      text= page.extract_text()

      if text:
        full_text +=text + "\n"

  return full_text


def extract_table(file_path):

  all_tables=[]

  with pdfplumber.open(file_path) as pdf:

    for page in pdf.pages:

      tables= page.extract_tables()

      if tables:
        all_tables.extend(tables)

  return all_tables


def extract_document(file_path):

  text=extract_text(file_path)
  tables=extract_table(file_path)

  combined_text= text + "\n\n"

  for i, table in enumerate(tables, start=1):

    combined_text += f"\n--- TABLE {i} ---\n"

    for row in table:

      clean_row = [
                cell if cell is not None else ""
                for cell in row
            ]
      combined_text += " | ".join(clean_row) + "\n"

    return combined_text
