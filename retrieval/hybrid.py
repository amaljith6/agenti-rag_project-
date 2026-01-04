from retrieval.bm25 import BM25Retriever
from retrieval.vector import vector_search
from retrieval.mmr import apply_mmr


class HybridRetriever:
    def __init__(self, vectorstore, documents, embedder):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.bm25 = BM25Retriever(documents)

    def retrieve(self, query: str, k: int = 5):
        # 1. BM25
        bm25_docs = self.bm25.search(query, k=k)

        # 2. FAISS
        faiss_docs = vector_search(self.vectorstore, query, k=k)

        # 3. Merge & deduplicate
        all_docs = {doc.page_content: doc for doc in bm25_docs + faiss_docs}
        merged_docs = list(all_docs.values())

        # 4. MMR
        query_emb = self.embedder.embed_query(query)
        doc_embs = self.embedder.embed_documents(
            [doc.page_content for doc in merged_docs]
        )

        final_docs = apply_mmr(
            query_embedding=query_emb,
            doc_embeddings=doc_embs,
            documents=merged_docs,
            k=k,
        )

        return final_docs
