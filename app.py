import streamlit as st
import os
from pathlib import Path
from main2 import LocalEmbedder, DocumentStore, Ingestor, Retriever, Reasoner, Synthesizer, export_markdown, export_pdf

# --------------------------- Setup ---------------------------
st.set_page_config(page_title="Deep Researcher Agent", layout="wide")
st.title("📄 Deep Researcher Agent")

# --------------------------- Sidebar -------------------------
st.sidebar.header("Settings")
index_folder = st.sidebar.text_input("Index Folder", "dr_index")
embedding_model_name = st.sidebar.text_input("Embedding Model", "all-MiniLM-L6-v2")
summarizer_model_name = st.sidebar.text_input("Summarizer Model (Optional)", "")

# --------------------------- Initialize Components ---------------------------
#@st.cache_resource(show_spinner=False)
def load_components():
    embedder = LocalEmbedder(embedding_model_name)
    dim = embedder.embed_texts(["hello"]).shape[1]
    store = DocumentStore(path_prefix=index_folder, dim=dim)
    synth = Synthesizer(model_name=summarizer_model_name or None)
    retriever = Retriever(embedder, store)
    reasoner = Reasoner(retriever, synth)
    return embedder, store, Ingestor(embedder, store), reasoner

embedder, store, ingestor, reasoner = load_components()

# --------------------------- File Upload ---------------------------
st.header("1️⃣ Upload Document(s)")
uploaded_files = st.file_uploader("Upload PDFs or text files", type=["pdf","txt","md","html"], accept_multiple_files=True)

if st.button("Ingest Uploaded Files"):
    if not uploaded_files:
        st.warning("Please upload at least one file to ingest.")
    else:
        for uploaded_file in uploaded_files:
            temp_path = Path("temp_uploads") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                ingestor.ingest(str(temp_path))
                st.success(f"✅ Ingested {uploaded_file.name}")
            except Exception as e:
                st.error(f"⚠️ Failed to ingest {uploaded_file.name}: {e}")

# --------------------------- Query Section ---------------------------
st.header("2️⃣ Ask a Question")
query = st.text_area("Enter your question here", height=100)
k = st.number_input("Number of top results to retrieve (k)", min_value=1, max_value=20, value=5)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving answer..."):
            result = reasoner.answer(query, k=k)
        st.subheader("✅ Final Answer")
        st.write(result['final'])

        st.subheader("Reasoning & Sub-steps")
        for step in result['substeps']:
            st.markdown(f"**Subquery:** {step['subquery']}")
            st.markdown(f"**Synthesized Answer:** {step['synthesized']}")
            st.markdown("**Evidence:**")
            for e in step['evidence']:
                st.markdown(f"- Source: {e.get('source')} (chunk {e.get('metadata', {}).get('chunk')})")
                st.markdown(f"  ```{e.get('text')[:500]}...```")

        # ------------------- Export Options -------------------
        st.header("3️⃣ Export Answer")
        md_path = st.text_input("Markdown file path", "report.md")
        pdf_path = st.text_input("PDF file path (requires pandoc)", "report.pdf")

        if st.button("Export Answer"):
            md_file = export_markdown(md_path, result)
            st.success(f"📄 Markdown exported: {md_file}")
            if pdf_path:
                try:
                    pdf_file = export_pdf(md_file, pdf_path)
                    st.success(f"📄 PDF exported: {pdf_file}")
                except Exception as e:
                    st.error(f"⚠️ PDF export failed: {e}")

st.sidebar.info("Deep Researcher Agent using local embeddings (SentenceTransformers + FAISS) and optional summarization. Note: add only the PDF files or TXT files that have text (1D).")
