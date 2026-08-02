from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils import get_embedding_model


def create_vector_db():

    loader = PyMuPDFLoader("sample.pdf")

    docs = loader.load()

    print(f"Pages Loaded : {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    print(f"Chunks Created : {len(chunks)}")

    embeddings = get_embedding_model()

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_db.save_local("vectorstore")

    print("Vector Database Saved Successfully!")


if __name__ == "__main__":
    create_vector_db()