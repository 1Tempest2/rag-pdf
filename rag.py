from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFacePipeline

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

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
llm = HuggingFacePipeline(pipeline=pipeline)

def upload_pdf(new_pdf):
    with open(pdf_directory + new_pdf.name, "wb") as f:
        f.write(new_pdf.getbuffer())


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_documents(documents):
    global vector_db
    chunks = text_split(documents)
    vector_db = FAISS.from_documents(chunks, embeddings)


def retrieve_documents(query, k = 3):
    return vector_db.similarity_search(query, k=k)


def answer_question(question, documents, llm):
    context = "\n\n".join([doc.page_content for doc in documents])
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdf_directory + uploaded_file.name)
    chunked_documents = text_split(documents)
    index_documents(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_documents(question)
        answer = answer_question(question, related_documents, llm)
        st.chat_message("assistant").write(answer)



