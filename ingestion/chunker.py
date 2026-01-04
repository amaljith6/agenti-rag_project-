from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
    )
    return splitter.split_documents(documents)
