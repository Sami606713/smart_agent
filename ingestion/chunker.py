from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Split a list of documents into smaller chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
