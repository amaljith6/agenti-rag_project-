from embeddings.indexer import load_index
from embeddings.embedder import get_embedding_model
from ingest import ingest
from retrieval.hybrid import HybridRetriever


class TourGuideRetrieverTool:
    def __init__(self):
        # Load FAISS index
        self.vectorstore = load_index("data/faiss/tour_guide")

        # Load documents for BM25
        self.documents = ingest(
            "data/raw_docs/tour_guide.pdf",
            domain="tour_guide",
        )

        self.embedder = get_embedding_model()

        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            documents=self.documents,
            embedder=self.embedder,
        )

    def run(self, query: str, k: int = 5):
        """
        Retrieve travel-related documents.
        """
        return self.retriever.retrieve(query, k=k)
