import streamlit as st
from pathlib import Path
from chunker import chunk_text
import pdf_processer as pdf
from gemma_answer_generator import RAGAnswerGenerator
from semantic_retriever import SemanticRetriever
from langchain.schema import Document


@st.cache_resource
def get_retriever():
    return SemanticRetriever()


@st.cache_resource
def get_model():
    return RAGAnswerGenerator()


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 RAG Assistant")

# Sidebar controls
with st.sidebar:
    # PDF upload
    uploaded_file = st.file_uploader("PDF feltöltése", type=["pdf"])

    # Model selection dropdown
    model_type = st.selectbox(
        "Modell kiválasztása",
        ["gemma", "open-ai"],
        index=0
    )

    # Debug toggle
    show_debug = st.checkbox("Debug nézet mutatása", value=False)

# Initialize components with selected model
retriever = get_retriever()
if(model_type=="gemma"):
    answering_model = get_model()
elif(model_type=="open-ai"):
    answering_model = get_model()

if uploaded_file is not None:
    if "current_document" not in st.session_state or st.session_state.current_document != uploaded_file.name:
        st.sidebar.info("Dokumentum feldolgozása...")
        pdf_path = pdf.upload_pdf(uploaded_file, "pdfs")
        raw_text = pdf.process_pdf(uploaded_file, "pdfs")
        chunks = [Document(page_content=chunk) for chunk in chunk_text(raw_text)]
        retriever.index_documents(chunks)
        st.session_state.chunks = chunks
        st.session_state.current_document = uploaded_file.name
    else:
        st.sidebar.success(f"✅ Dokumentum már feldolgozva ({len(st.session_state.chunks)} chunk)")

# Chat interface
query = st.text_input("Kérdezz valamit:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Információ keresése..."):
        related_documents = retriever.retrieve(query)
        retrieved_documents = [{"chunk": doc.page_content} for doc in related_documents]
        st.session_state.retrieved_documents = retrieved_documents
        answer = answering_model.generate_answer(query, retrieved_documents)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)
else:
    st.warning("Először tölts fel egy PDF-et!")

# Debug view
if show_debug:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧱 Chunk list")
        if "chunks" in st.session_state and st.session_state.chunks:
            for i, chunk in enumerate(st.session_state.chunks, 1):
                with st.expander(f"Szegmens {i}", expanded=False):
                    st.markdown(
                        f"<div style='background-color: #0E1117; color: #ffffff; padding: 10px; border-radius: 5px; margin: 5px 0;'>"
                        f"{chunk.page_content}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.warning("❗ Nincs feldolgozott dokumentum.")

    with col2:
        st.subheader("🤖 Retrieved Docs")
        if "retrieved_documents" in st.session_state and st.session_state.retrieved_documents:
            for i, retrieved_doc in enumerate(st.session_state.retrieved_documents, 1):
                with st.expander(f"Retrieved {i}", expanded=False):
                    st.markdown(
                        f"<div style='background-color: #0E1117; color: #ffffff; padding: 10px; border-radius: 5px; margin: 5px 0;'>"
                        f"{retrieved_doc['chunk']}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.warning("❗ Még nincs retrieved dokumentum.")