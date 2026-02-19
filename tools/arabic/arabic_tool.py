"""
Core Knowledge Tools Module

Provides semantic search and retrieval functionality over the knowledge base.
These tools are available to agents for information retrieval.
"""

import json
from typing import Optional
from langchain.tools import tool, ToolRuntime
from store.store import StoreService
import uuid


@tool
async def search_arabic_store(
    query: str,
    k: int = 5,
    runtime: ToolRuntime = None
) -> str:
    """
    Perform semantic search over stored documents in your knowledge base.
    
    Use this to find relevant information that has been previously stored.

    Args:
        query: Natural language query string to search for relevant documents.
        k: Number of top documents to return (default: 5).

    Returns:
        A formatted string with document IDs and text content.
    """
    
    if not runtime:
        return "Error: Runtime not available. Tool must be called within agent context."
    
    try:
        service = StoreService()
        results = await service.search(
            namespace=("knowledge_arabic",),
            query=query,
            limit=k
        )
        
        if not results:
            return f"No relevant documents found for query: {query}"
        
        # Extract ID and text content only
        content_with_ids = []
        all_docs = []
        
        for result in results:
            if hasattr(result, "key") and hasattr(result, "value"):
                key = result.key
                value = result.value
            else:
                # Fallback for different return types
                key = getattr(result, "key", "unknown")
                value = getattr(result, "value", {})
            
            # Extract text
            if isinstance(value, dict):
                text = value.get("text") or value.get("chunk") or ""
            else:
                text = str(value)        
            
            content_with_ids.append(f"[Document ID: {key}]\n{text}")
            all_docs.append({"id": key, "text": text})
        
        # Update state
        runtime.state["last_search_results"] = all_docs
        runtime.state["search_query"] = query
        runtime.state["search_count"] = len(all_docs)
        
        return "\n\n---\n\n".join(content_with_ids)
    
    except Exception as e:
        return f"Error searching store: {str(e)}"


@tool
async def retrieve_arabic_document(
    document_key: str,
    runtime: ToolRuntime = None
) -> str:
    """
    Retrieve a specific document from the knowledge base by its key.
    
    Use this after search_store to get the full content of a specific document.

    Args:
        document_key: The unique key/ID of the document to retrieve.

    Returns:
        The document ID and full text content.
    """
    
    if not runtime:
        return "Error: Runtime not available."
    
    try:
        service = StoreService()
        result = await service.get(
            namespace=("knowledge_arabic",),
            key=document_key
        )
        
        if not result:
            return f"Document '{document_key}' not found"
        
        # result is likely a StoreItem
        value = result.value if hasattr(result, 'value') else result
        
        # Extract text
        if isinstance(value, dict):
            text = value.get("text") or value.get("chunk") or ""
        else:
            text = str(value)
        
        # Update state
        runtime.state["last_retrieved_doc"] = document_key
        
        return f"[Document ID: {document_key}]\n{text}"
    
    except Exception as e:
        return f"Error retrieving document: {str(e)}"


@tool
async def list_arabic_stored_documents(
    limit: int = 10,
    runtime: ToolRuntime = None
) -> str:
    """
    List all stored documents in the knowledge base.
    
    Useful for discovering what content is available.

    Args:
        limit: Maximum number of documents to list (default: 10).

    Returns:
        A list of document IDs and brief content snippets.
    """
    
    if not runtime:
        return "Error: Runtime not available."
    
    try:
        service = StoreService()
        items = await service.list_items(
            namespace=("knowledge_arabic",),
            limit=limit
        )
        
        if not items:
            return "No documents currently stored"
        
        doc_list = []
        for item in items:
            key = item.key if hasattr(item, 'key') else "unknown"
            value = item.value if hasattr(item, 'value') else {}
            
            # Extract text
            if isinstance(value, dict):
                text = value.get("text") or value.get("chunk") or ""
            else:
                text = str(value)
            
            preview = text[:100] + "..." if len(text) > 100 else text
            doc_list.append(f"[Document ID: {key}]\n{preview}")
            
        # Update state
        runtime.state["listed_documents_count"] = len(doc_list)
        
        return "\n\n---\n\n".join(doc_list)
    
    except Exception as e:
        return f"Error listing documents: {str(e)}"


def get_arabic_knowledge_tools():
    """Returns all core knowledge retrieval tools."""
    return [search_arabic_store, retrieve_arabic_document, list_arabic_stored_documents]


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    class MockRuntime:
        def __init__(self, state):
            self.state = state

    async def test_search():
        print("--- Testing search_store tool ---")
        
        query = "ما هو دور المعلم؟" # "What is the role of the teacher?"
        print(f"Query: {query}")
        
        result = await search_store.ainvoke(
            {"query": query, "k": 3},
            {"runtime": mock_runtime}
        )
        
        print("\nSearch Result:")
        print(result)

    asyncio.run(test_search())