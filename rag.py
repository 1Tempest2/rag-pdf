from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import transformers
import torch

load_dotenv()

pdf_directory = "./pdfs"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
    text_splitter = CharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        add_start_index=True
    )



