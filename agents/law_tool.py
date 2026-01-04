from embeddings.indexer import load_index
from embeddings.embedder import get_embedding_model
from ingest import ingest
from retrieval.hybrid import HybridRetriever


class LawRetrieverTool:
    def __init__(self):
        # Load FAISS index
        self.vectorstore = load_index("data/faiss/law")

        # Load documents for BM25
        self.documents = ingest(
            "data/raw_docs/indian_constitution.pdf",
            domain="law",
        )

        self.embedder = get_embedding_model()

        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            documents=self.documents,
            embedder=self.embedder,
        )

    def run(self, query: str, k: int = 5):
        """
        Retrieve legal documents.
        """
        return self.retriever.retrieve(query, k=k)
