from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

texts = [
    "RAG stands for Retrieval Augmented Generation.",
    "FAISS is a vector database.",
    "LLaMA is an open-source LLM."
]

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"}
)

db = FAISS.from_texts(texts, embeddings)

docs = db.similarity_search("What is RAG?", k=2)

for d in docs:
    print(d.page_content)
