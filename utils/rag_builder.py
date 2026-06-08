from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


def build_vector_store(texts_dict):

  text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
  )

  documents=[]

  for doc_name, text in texts_dict.items():

    chunks=text_splitter.create_documents(
      [text],
      metadatas=[{"source": doc_name}]
    )

    documents.extend(chunks)


  embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small"
  )

  vectore_store=FAISS.from_documents(
    documents,
    embeddings
  )

  return vectore_store.as_retriever(
    search_kwargs={"k": 4}
  )


def get_relevant_chunks(query, retriever):

  docs= retriever.invoke(query)

  return docs
