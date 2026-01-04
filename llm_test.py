from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3:8b")

print(llm.invoke("Explain FAISS in one sentence.").content)
