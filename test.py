import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

prompt_template = PromptTemplate(
input_variables=["context", "question"],
template="""
A kérdésre CSAK az alábbi szövegrészletek alapján válaszolj pontosan és lényegre törően. 
Csak olyan információt használj, ami a szövegrészletekből egyértelműen kiderül. 
Ha nincs elegendő információ a válaszhoz, mondd azt, hogy Nem található pontos válasz a megadott dokumentumok alapján.
Kérdés: {question} 
Kontextus: {context} 
Answer:
""")

pdf_directory = 'pdfs/'

embeddings = HuggingFaceEmbeddings(model_name="NYTK/sentence-transformers-experimental-hubert-hungarian")


llm = OllamaLLM(model="deepseek-r1:14b")

def upload_pdf(new_pdf):
    with open(pdf_directory + "/" + new_pdf.name, "wb") as f:
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