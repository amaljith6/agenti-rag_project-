import os
from langchain_community.vectorstores import FAISS
from embeddings.embedder import get_embedding_model


def build_and_save_index(documents, index_dir: str):
    """
    Builds a FAISS index from documents and saves it locally.
    """
    os.makedirs(index_dir, exist_ok=True)

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    vectorstore.save_local(index_dir)

    return vectorstore


def load_index(index_dir: str):
    """
    Loads a FAISS index from disk.
    """
    embeddings = get_embedding_model()

    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
