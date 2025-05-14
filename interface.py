import streamlit as st
from chunker import chunk_text
import pdf_processer as pdf

st.set_page_config(page_title="RAG Chat Interface", layout="wide")

# Style
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user {
        background-color: #DCF8C6;
        text-align: right;
    }
    .bot {
        background-color: #F1F0F0;
        text-align: left;
    }
    .chunk-box {
        background-color: #262730;
        padding: 1em;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 0.5em;
    }
    
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📄 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Upload feedback
if uploaded_file is not None:
    st.sidebar.success("✅ PDF uploaded successfully!")
    preprocessed_text = pdf.process_pdf(uploaded_file, "pdfs")
    chunks = chunk_text(preprocessed_text)
    st.session_state.chunks = chunks
else:
    st.sidebar.info("Upload a PDF file to get started.")

# Main Title
st.title("🤖")

st.subheader("💬 Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:", key="chat_input")
if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", "🤖 [Response will appear here...]"))

for role, message in reversed(st.session_state.chat_history):
    with st.container():
        role_class = "user" if role == "user" else "bot"
        st.markdown(f'<div class="chat-message {role_class}">{message}</div>', unsafe_allow_html=True)

st.subheader("🧱 Chunk Explorer")

col1, col2 = st.columns(2)

with col1:
    if st.button("List All Chunks"):
        if "chunks" in st.session_state:
            st.markdown("### 📚 All Chunks")
            for i, chunk in enumerate(st.session_state.chunks):
                st.markdown(f"<div class='chunk-box'>Chunk {i+1}: {chunk}</div>", unsafe_allow_html=True)
        else:
            st.warning("❗ Please upload a PDF first.")

with col2:
    if st.button("Show Retrieved Chunks"):
        # Placeholder for retrieval result
        st.markdown("### 🔍 Retrieved Chunks")
        for i in range(2):
            st.markdown(f"<div class='chunk-box'>Retrieved Chunk {i+1}: Matching content from document...</div>", unsafe_allow_html=True)
