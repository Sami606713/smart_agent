import uuid
from typing import Optional, Any

from langgraph.store.postgres import AsyncPostgresStore
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

class StoreService:
    """
    Service class for managing persistent vector storage using LangGraph's AsyncPostgresStore.
    Handles knowledge storage with language-specific namespacing.
    """
    
    def __init__(self):
        self._store: Optional[AsyncPostgresStore] = None
        self._db_uri = self._get_db_uri()
    
    @staticmethod
    def _get_db_uri() -> str:
        """
        Get the database URI from environment variable 'postgress_url'.
        """
        uri = os.getenv("POSTGRES_URI")
        if not uri:
            raise ValueError("Environment variable 'POSTGRES_URI' is not set")
        
        # Convert SQLAlchemy async format to standard PostgreSQL format if necessary
        if uri.startswith("postgresql+asyncpg://"):
            uri = uri.replace("postgresql+asyncpg://", "postgresql://", 1)
        elif uri.startswith("postgresql+psycopg://"):
            uri = uri.replace("postgresql+psycopg://", "postgresql://", 1)
        
        return uri
    
    @staticmethod
    def _get_embeddings():
        """Get Sentence Transformer embeddings instance (dims=384)."""
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    async def _get_embed_func(self):
        """Create the embedding function for the store."""
        embeddings = self._get_embeddings()
        
        async def embed_texts(texts: list[str]) -> list[list[float]]:
            # HuggingFaceEmbeddings.aembed_documents is typically available
            return await embeddings.aembed_documents(texts)
        
        return embed_texts
    
    async def _create_store(self) -> AsyncPostgresStore:
        """Factory to create a configured store instance."""
        embed_func = await self._get_embed_func()
        
        return AsyncPostgresStore.from_conn_string(
            self._db_uri,
            index={
                "dims": 384,
                "embed": embed_func,
                "fields": ["text"]  # Fields to vectorize in the JSON data
            }
        )
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
    async def save(
        self,
        namespace: tuple[str, ...],
        content: str,
        metadata: dict | None = None
    ) -> str:
        """
        Save content to a specific namespace.
        
        Args:
            namespace: The namespace tuple (e.g., ("knowledge_arabic",))
            content: The text content to store
            metadata: Optional metadata dict
            
        Returns:
            str: The generated item ID
        """
        async with await self._create_store() as store:
            item_id = str(uuid.uuid4())
            
            await store.aput(
                namespace,
                item_id,
                {
                    "text": content,
                    "metadata": metadata or {"source": "manual"}
                }
            )
            print(f"Stored item {item_id} in namespace {namespace}")
            return item_id
    
    async def save_knowledge_batch(
        self,
        namespace: tuple[str, ...],
        contents: list[dict],
        source: str = "files"
    ) -> list[str]:
        """
        Save multiple knowledge items in batch.
        
        Args:
            namespace: The namespace tuple
            contents: List of dicts with 'text' and optional 'metadata' keys
            source: Source type (e.g., "web", "file", "youtube")
        
        Returns:
            list[str]: List of generated item IDs
        """
        item_ids = []
        
        # Build base metadata
        base_metadata = {
            "source": source,
        }
        
        async with await self._create_store() as store:
            for item in contents:
                item_id = str(uuid.uuid4())
                
                # Merge base metadata with item-specific metadata
                full_metadata = {**base_metadata}
                if item.get("metadata"):
                    full_metadata.update(item["metadata"])
                
                await store.aput(
                    namespace,
                    item_id,
                    {
                        "text": item["text"],
                        "metadata": full_metadata
                    }
                )
                item_ids.append(item_id)
            
            print(f"Stored {len(item_ids)} knowledge items in namespace {namespace}")
            return item_ids
    
    async def get(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> Optional[Any]:
        """
        Get a specific item from the store by key.
        
        Args:
            namespace: The namespace tuple
            key: The item ID/key
            
        Returns:
            The stored item or None if not found
        """
        async with await self._create_store() as store:
            result = await store.aget(namespace, key)
            return result


    async def delete(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> None:
        """
        Delete a specific item from the store.
        
        Args:
            namespace: The namespace tuple
            key: The item ID/key to delete
        """
        async with await self._create_store() as store:
            await store.adelete(namespace, key)
            print(f"Deleted item {key} from namespace {namespace}")

    
    async def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        filter: dict | None = None,
        limit: int = 10
    ) -> list[Any]:
        """
        Search for items in a namespace using semantic search.
        
        Args:
            namespace: The namespace tuple
            query: The search query
            filter: Optional dictionary to filter by metadata
            limit: Maximum number of results to return
            
        Returns:
            List of matching items
        """
        async with await self._create_store() as store:
            results = await store.asearch(namespace, query=query, filter=filter, limit=limit)
            return results
    
    
    async def list_items(
        self,
        namespace: tuple[str, ...],
        limit: int = 100,
        offset: int = 0
    ) -> list[Any]:
        """
        List all items in a namespace.
        
        Args:
            namespace: The namespace tuple
            limit: Maximum number of items to return
            offset: Number of items to skip
            
        Returns:
            List of items in the namespace
        """
        async with await self._create_store() as store:
            results = await store.alist(namespace, limit=limit, offset=offset)
            return results
    
    async def setup(self) -> None:
        """Initialize the store (creates tables if needed)."""
        async with await self._create_store() as store:
            await store.setup()
            print("✅ Store setup complete!")


# =========================================================================
# Singleton instance and convenience functions for backward compatibility
# =========================================================================

if __name__ == "__main__":
    async def main():
        service = StoreService()
        await service.setup()
    
    asyncio.run(main())
