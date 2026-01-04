def vector_search(vectorstore, query: str, k: int = 5):
    """
    Semantic search using FAISS.
    """
    return vectorstore.similarity_search(query, k=k)
