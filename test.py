import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

template = """
A kérdésre CSAK az alábbi szövegrészletek alapján válaszolj pontosan és lényegre törően. 
Csak olyan információt használj, ami a szövegrészletekből egyértelműen kiderül. 
Ha nincs elegendő információ a válaszhoz, mondd azt, hogy Nem található pontos válasz a megadott dokumentumok alapján.
Kérdés: {question} 
Kontextus: {context} 
Answer:
"""

pdfs_directory = 'pdfs/'

embeddings = HuggingFaceEmbeddings(model_name="NYTK/sentence-transformers-experimental-hubert-hungarian")


model = OllamaLLM(model="deepseek-r1:14b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

def index_documents(documents):
    global vector_db
    chunks = split_text(documents)
    vector_db = FAISS.from_documents(chunks, embeddings)


def retrieve_documents(query, k = 3):
    return vector_db.similarity_search(query, k=k)


def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_documents(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_documents(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)