from ingestion.loader import load_pdf
from ingestion.normalizer import normalize_text
from ingestion.enrich import enrich
from ingestion.chunker import chunk_documents
from langchain_core.documents import Document


def ingest(pdf_path: str, domain: str):
    elements = load_pdf(pdf_path)

    enriched_docs = enrich(
        elements=elements,
        domain=domain,
        source=pdf_path,
    )

    normalized_docs = [
        Document(
            page_content=normalize_text(doc.page_content),
            metadata=doc.metadata,
        )
        for doc in enriched_docs
    ]

    chunks = chunk_documents(normalized_docs)
    return chunks


if __name__ == "__main__":
    tour_chunks = ingest(
        "data/raw_docs/tour_guide.pdf",
        domain="tour_guide",
    )

    law_chunks = ingest(
        "data/raw_docs/indian_constitution.pdf",
        domain="law",
    )

    print(f"Tour guide chunks: {len(tour_chunks)}")
    print(f"Law chunks: {len(law_chunks)}")
