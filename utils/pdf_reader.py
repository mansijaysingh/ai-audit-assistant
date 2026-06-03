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