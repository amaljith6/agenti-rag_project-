from embeddings.indexer import load_index
from embeddings.embedder import get_embedding_model
from ingest import ingest
from retrieval.hybrid import HybridRetriever


# Load index
vectorstore = load_index("data/faiss/law")

# Load raw docs again (BM25 needs text corpus)
docs = ingest("data/raw_docs/indian_constitution.pdf", "law")

embedder = get_embedding_model()

retriever = HybridRetriever(
    vectorstore=vectorstore,
    documents=docs,
    embedder=embedder,
)

results = retriever.retrieve(
    "What is Article 21?",
    k=3,
)

for r in results:
    print("----")
    print(r.page_content)
    print(r.metadata)
