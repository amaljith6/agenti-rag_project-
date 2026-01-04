from embeddings.indexer import load_index

tour_index = load_index("data/faiss/tour_guide")
law_index = load_index("data/faiss/law")

print("Tour index size:", tour_index.index.ntotal)
print("Law index size:", law_index.index.ntotal)
