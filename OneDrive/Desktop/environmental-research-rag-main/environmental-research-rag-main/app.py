import streamlit as st
from rag_pipeline import build_rag

st.title("Environmental Research Assistant (RAG)")

uploaded_file = st.file_uploader("Upload research paper", type="pdf")

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    qa = build_rag("temp.pdf")

    question = st.text_input("Ask a question about the document")

    if question:
        answer = qa.run(question)
        st.write(answer)
