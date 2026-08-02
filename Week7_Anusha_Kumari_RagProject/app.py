import streamlit as st
from rag import RAG

st.set_page_config(
    page_title="Document Question Answering System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System (RAG)")

st.write("Upload a PDF and ask questions about it.")

if "rag" not in st.session_state:
    st.session_state.rag = None

if st.button("Load Vector Database"):

    st.session_state.rag = RAG()

    st.success("Vector database loaded successfully!")

question = st.text_input("Ask a question")

if st.button("Get Answer"):

    if st.session_state.rag is None:

        st.warning("Please load the vector database first.")

    elif question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching..."):

            answer = st.session_state.rag.ask(question)

        st.subheader("Answer")

        st.write(answer)