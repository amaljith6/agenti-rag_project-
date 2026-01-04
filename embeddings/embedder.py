from langchain_ollama import OllamaEmbeddings


def get_embedding_model():
    """
    Returns an embedding model served by Ollama.
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",  # small, fast, CPU-friendly
        base_url="http://localhost:11434",
    )
