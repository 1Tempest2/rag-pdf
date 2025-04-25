from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
import transformers
import torch

load_dotenv()


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
A kérdésre CSAK az alábbi szövegrészletek alapján válaszolj pontosan és lényegre törően. 
Csak olyan információt használj, ami a szövegrészletekből egyértelműen kiderül. 
Ha nincs elegendő információ a válaszhoz, mondd azt, hogy "Nem található pontos válasz a megadott dokumentumok alapján."

### Kérdés:
{question}

### Válasz:
""",
)

pdf_directory = "./pdfs"
embeddings = HuggingFaceEmbeddings(model_name="NYTK/sentence-transformers-experimental-hubert-hungarian")
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit":True},
    device_map="auto",
)

def upload_pdf(new_pdf):
    with open(pdf_directory + new_pdf.name, "wb") as f:
        f.write(new_pdf.getbuffer())


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_documents(documents):
    global vector_db
    chunks = text_splitter(documents)
    vector_db = FAISS.from_documents(chunks, embeddings)


def retrieve_documents(query, k = 3):
    return vector_db.similarity_search(query, k=k)


def answer_question(question, documents, llm):
    context = "\n\n".join([doc.page_content for doc in documents])
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})



