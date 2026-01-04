import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance


def apply_mmr(
    query_embedding,
    doc_embeddings,
    documents,
    k: int = 5,
    lambda_mult: float = 0.5,
):
    """
    Apply Maximal Marginal Relevance (MMR) to select diverse documents.
    """

    # Convert lists → numpy arrays (REQUIRED)
    query_embedding = np.array(query_embedding)
    doc_embeddings = np.array(doc_embeddings)

    idxs = maximal_marginal_relevance(
        query_embedding=query_embedding,
        embedding_list=doc_embeddings,
        k=k,
        lambda_mult=lambda_mult,
    )

    return [documents[i] for i in idxs]
