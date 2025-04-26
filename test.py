from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

import streamlit as st

if 'config' not in st.session_state:
    st.session_state.config = {
        'selected_model': "deepseek-r1:14b",
        'selected_embedding': "NYTK/sentence-transformers-experimental-hubert-hungarian",
        'chunk_size': 1024,
        'chunk_overlap': 256,
        'top_k': 4
    }

prompt_template = PromptTemplate(
input_variables=["context", "question"],
template="""
Kérlek, a következő kérdésre KIZÁRÓLAG a mellékelt szövegrészletek alapján MAGYARUL válaszolj.
Pontosság: Csak olyan információt használj, ami a szövegrészletekből egyértelműen levezethető.
Hiányos információ: Ha a válasz nem teljesen pontos, de a kontextusból következtetni lehet egy hozzávetőleges válaszra, add meg azt, de jelöld egyértelműen, hogy ez csak becslés.
Nincs válasz: Ha a szövegrészletek nem tartalmaznak releváns információt, válaszolj: „Nem található pontos válasz a megadott dokumentumok alapján.”
Formátum:
    Válasz rövid, lényegretörő, tételes vagy egyszerű mondatokban.
Kérdés: {question}
Kontextus: {context}
Válasz:
""")

pdf_directory = 'pdfs/'

embeddings = HuggingFaceEmbeddings(model_name=st.session_state.config['selected_embedding'])
answering_model = OllamaLLM(model=st.session_state.config['selected_model'])

def upload_pdf(new_pdf):
    with open(pdf_directory + "/" + new_pdf.name, "wb") as f:
        f.write(new_pdf.getbuffer())


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.config['chunk_size'],
        chunk_overlap=st.session_state.config['chunk_overlap'],
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_documents(documents):
    global vector_db
    chunks = text_split(documents)
    vector_db = FAISS.from_documents(chunks, embeddings)


def retrieve_documents(query, k=st.session_state.config['top_k']):
    return vector_db.similarity_search(query, k=k)


def answer_question(question, documents, llm):
    context = "\n\n".join([doc.page_content for doc in documents])
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})



st.set_page_config(
    page_title="RAG asszisztens",
    page_icon="🤖",
    layout="wide"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .configheader {
    text-align: center;
    font-weight: bold;    
    font-size: 3rem;
    color: #3498db;       
    font-family: 'Arial', sans-serif; 
}
    </style>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("<h3 class='configheader'>Config</h3>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Tölts fel egy PDF dokumentumot",
        type="pdf",
        accept_multiple_files=False,
        help="Válassz ki egy PDF dokumentumot"
    )

    available_models = [
        "deepseek-r1:14b"
    ]

    st.session_state.config['selected_model'] = st.selectbox(
        "Válasz ki egy LLM-t",
        options=available_models,
        index=available_models.index(st.session_state.config['selected_model']),
        help="Válaszd ki, hogy melyik LLM segitsen válaszolni!"
    )

    embedding_models = [
        "NYTK/sentence-transformers-experimental-hubert-hungarian",
    ]

    st.session_state.config['selected_embedding'] = st.selectbox(
        "Beágyazó model választás",
        options=embedding_models,
        index=embedding_models.index(st.session_state.config['selected_embedding'])
    )

    with st.expander("Egyéb beállítások"):
        st.session_state.config['chunk_size'] = st.slider(
            "Chunk Size", 512, 2048, st.session_state.config['chunk_size'], 64
        )
        st.session_state.config['chunk_overlap'] = st.slider(
            "Chunk Overlap", 0, 512, st.session_state.config['chunk_overlap'], 64
        )
        st.session_state.config['top_k'] = st.slider(
            "K dokumentum", 1, 10, st.session_state.config['top_k'], 1
        )

    st.markdown("---")

main_col1, main_col2 = st.columns([3, 2])

with main_col1:
    st.markdown("### Itt tedd fel a kérdéseidet!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if uploaded_file:
        upload_pdf(uploaded_file)
        documents = load_pdf(pdf_directory + uploaded_file.name)
        chunked_documents = text_split(documents)
        index_documents(chunked_documents)

        question = st.chat_input("Kérdezz valamit a dokumentumról!")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.write(question)

            with st.spinner("Információ keresése..."):
                related_documents = retrieve_documents(question)
                answer = answer_question(question, related_documents, answering_model)

            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.write(answer)
    else:
        st.info("Kérlek tölts fel egy PDF filet!")

with main_col2:
    st.markdown("### PDF információi")

    if uploaded_file:
        st.info("PDF sikeresen feltöltve")

        st.metric("Oldalak", len(documents))
        st.metric("Szöveg chunkok", len(chunked_documents))

        with st.expander("Chunk preview"):
            if len(chunked_documents) >= 5:
                chunk_preview = chunked_documents[:5]
            else:
                chunk_preview = chunked_documents[:len(chunked_documents)]
            st.write(chunk_preview)
        if "messages" in st.session_state and len(st.session_state.messages) > 0 and st.session_state.messages[-1][
            "role"] == "assistant":
            st.markdown("### Top találatok:")
            with st.expander("Top K Retrieved"):
                st.write(related_documents)
    else:
        st.warning("Nincs betöltött dokumentum!")


st.markdown("---")