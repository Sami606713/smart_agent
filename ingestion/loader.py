import os
from pathlib import Path
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.documents import Document

def load_documents_from_dir(base_dir: str) -> list[Document]:
    """
    Load all PDF documents from subdirectories within base_dir.
    Metadata includes 'folder_name' to identify language.
    """
    documents = []
    base_path = Path(base_dir)
    
    # Iterate through subdirectories (e.g., english, arabic)
    for sub_dir in base_path.iterdir():
        if sub_dir.is_dir():
            folder_name = sub_dir.name
            print(f"Loading documents from folder: {folder_name}")
            
            for file_path in sub_dir.glob("*.pdf"):
                print(f"  Loading: {file_path.name}")
                loader = PyMuPDF4LLMLoader(str(file_path))
                loaded_docs = loader.load()
                
                # Add folder_name and custom language flag to metadata
                for doc in loaded_docs:
                    doc.metadata["folder_name"] = folder_name
                    doc.metadata["language"] = "arabic" if "arabic" in folder_name.lower() else "english"
                    doc.metadata["file_name"] = file_path.name
                
                documents.extend(loaded_docs)
                
    return documents
