from ingest import ingest
from embeddings.indexer import build_and_save_index


if __name__ == "__main__":
    # Tour Guide
    tour_chunks = ingest(
        "data/raw_docs/tour_guide.pdf",
        domain="tour_guide",
    )

    build_and_save_index(
        documents=tour_chunks,
        index_dir="data/faiss/tour_guide",
    )

    # Law
    law_chunks = ingest(
        "data/raw_docs/indian_constitution.pdf",
        domain="law",
    )

    build_and_save_index(
        documents=law_chunks,
        index_dir="data/faiss/law",
    )

    print("✅ FAISS indexes built and saved")
