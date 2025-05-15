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

retriever = get_retriever()
answering_model = get_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 Magyar Dokumentum Kereső")

# PDF upload sidebar
uploaded_file = st.sidebar.file_uploader("PDF feltöltése", type=["pdf"])

if uploaded_file is not None:
    # Only process if new document uploaded
    if "current_document" not in st.session_state or st.session_state.current_document != uploaded_file.name:
        st.sidebar.info("Dokumentum feldolgozása...")

        # Save and process PDF text (assuming these functions exist)
        pdf_path = pdf.upload_pdf(uploaded_file, "pdfs")
        raw_text = pdf.process_pdf(uploaded_file, "pdfs")

        # Chunk text
        chunks = [Document(page_content=chunk) for chunk in chunk_text(raw_text)]

        # Add chunks to retriever
        retriever.index_documents(chunks)

        # Store in session state
        st.session_state.chunks = chunks
        st.session_state.current_document = uploaded_file.name
        st.sidebar.success(f"✅ {len(chunks)} szegmens feldolgozva!")

    else:
        st.sidebar.success(f"✅ Dokumentum már feldolgozva ({len(st.session_state.chunks)} szegmens)")

# Query input
query = st.text_input("Kérdezz valamit:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.spinner("Információ keresése..."):
        related_documents = retriever.retrieve(query)
        retrieved_documents = [{"chunk": doc.page_content} for doc in related_documents]
        answer = answering_model.generate_answer(query, retrieved_documents)

        st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
else:
    st.warning("Először tölts fel egy PDF-et!")

# Debug toggle
show_debug = st.sidebar.checkbox("Debug nézet mutatása", value=False)

if show_debug:
    st.subheader("🧱 Szegmensek listája")
    if "chunks" in st.session_state and st.session_state.chunks:
        # Display all chunks as a numbered list
        for i, chunk in enumerate(st.session_state.chunks, 1):
            st.markdown(f"**Szegmens {i}:** {chunk}")
    else:
        st.warning("❗ Nincs feldolgozott dokumentum.")

# Optional: clear all session state and retriever
if st.sidebar.button("Új dokumentum, törlés"):
    st.session_state.clear()
    retriever = get_retriever()
