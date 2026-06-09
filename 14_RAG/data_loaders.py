import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import (TextLoader, WebBaseLoader, DirectoryLoader, PyPDFLoader)

from dotenv import load_dotenv
load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file.write(b"Hello, This is sample text file. \n This file will be used to study RAG. \n Line Third for example")
        temp_file_path = temp_file.name
    
    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(doc)
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)


def doc_structure():
    pass


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    print(documents)
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}")
        print(f"MetaData: {doc.metadata}")


if __name__ == "__main__":
    load_text_file()
    print("="*50)
    pdf_loader('./docs/RAG_Overview.pdf')