"""
Core Knowledge Tools Module for English
"""

import json
from typing import Optional
from langchain.tools import tool, ToolRuntime
from store.store import StoreService
import uuid

@tool
async def search_english_store(
    query: str,
    k: int = 5,
    runtime: ToolRuntime = None
) -> str:
    """
    Perform semantic search over stored English documents in your knowledge base.
    
    Args:
        query: Natural language query string.
        k: Number of top documents to return.
    """
    if not runtime:
        return "Error: Runtime not available."
    
    try:
        service = StoreService()
        results = await service.search(
            namespace=("knowledge_eng",),
            query=query,
            limit=k
        )
        
        if not results:
            return f"No relevant English documents found for query: {query}"
        
        content_with_ids = []
        all_docs = []
        
        for result in results:
            key = getattr(result, "key", "unknown")
            value = getattr(result, "value", {})
            
            if isinstance(value, dict):
                text = value.get("text") or value.get("chunk") or ""
            else:
                text = str(value)        
            
            content_with_ids.append(f"[Document ID: {key}]\n{text}")
            all_docs.append({"id": key, "text": text})
        
        runtime.state["last_english_search_results"] = all_docs
        
        return "\n\n---\n\n".join(content_with_ids)
    
    except Exception as e:
        return f"Error searching English store: {str(e)}"

@tool
async def retrieve_english_document(
    document_key: str,
    runtime: ToolRuntime = None
) -> str:
    """
    Retrieve a specific English document from the knowledge base by its key.
    """
    if not runtime:
        return "Error: Runtime not available."
    
    try:
        service = StoreService()
        result = await service.get(
            namespace=("knowledge_eng",),
            key=document_key
        )
        
        if not result:
            return f"English document '{document_key}' not found"
        
        value = result.value if hasattr(result, 'value') else result
        
        if isinstance(value, dict):
            text = value.get("text") or value.get("chunk") or ""
        else:
            text = str(value)
        
        runtime.state["last_retrieved_eng_doc"] = document_key
        
        return f"[Document ID: {document_key}]\n{text}"
    
    except Exception as e:
        return f"Error retrieving English document: {str(e)}"

@tool
async def list_english_stored_documents(
    limit: int = 10,
    runtime: ToolRuntime = None
) -> str:
    """
    List all stored English documents in the knowledge base.
    """
    if not runtime:
        return "Error: Runtime not available."
    
    try:
        service = StoreService()
        items = await service.list_items(
            namespace=("knowledge_eng",),
            limit=limit
        )
        
        if not items:
            return "No English documents currently stored"
        
        doc_list = []
        for item in items:
            key = item.key if hasattr(item, 'key') else "unknown"
            value = item.value if hasattr(item, 'value') else {}
            
            if isinstance(value, dict):
                text = value.get("text") or value.get("chunk") or ""
            else:
                text = str(value)
            
            preview = text[:100] + "..." if len(text) > 100 else text
            doc_list.append(f"[Document ID: {key}]\n{preview}")
            
        runtime.state["listed_english_documents_count"] = len(doc_list)
        
        return "\n\n---\n\n".join(doc_list)
    
    except Exception as e:
        return f"Error listing English documents: {str(e)}"

def get_eng_knowledge_tools():
    """Returns all English knowledge retrieval tools."""
    return [search_english_store, retrieve_english_document, list_english_stored_documents]
