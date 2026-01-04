''' modules'''

from langchain_core.documents import Document


def enrich(elements, domain: str, source: str):
    """
    Convert unstructured elements into LangChain Documents
    with metadata.
    """
    docs = []

    for el in elements:
        if not hasattr(el, "text"):
            continue

        docs.append(
            Document(
                page_content=el.text,
                metadata={
                    "domain": domain,
                    "source": source,
                    "content_type": "legal" if domain == "law" else "travel",
                },
            )
        )

    return docs
