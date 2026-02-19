import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.loader import load_documents_from_dir
from ingestion.chunker import split_documents
from store.store import StoreService

async def ingest_pipeline():
    # 1. Setup base data directory
    base_data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(base_data_dir):
        print(f"Error: Data directory {base_data_dir} not found.")
        return

    # 2. Load documents
    print("--- Starting Document Loading ---")
    documents = load_documents_from_dir(base_data_dir)
    print(f"Total documents loaded (pages): {len(documents)}")

    if not documents:
        print("No documents found to ingest.")
        return

    # 3. Split documents into chunks
    print("\n--- Starting Document Splitting ---")
    # User requested chunk_size=100, chunk_overlap=20
    chunked_docs = split_documents(documents, chunk_size=100, chunk_overlap=20)
    print(f"Total chunks created: {len(chunked_docs)}")

    # 4. Initialize StoreService and Ingest to Postgres
    print("\n--- Starting Ingestion to Postgres Store ---")
    store_service = StoreService()
    
    # Ensure database tables are created
    await store_service.setup()
    
    # Group chunks by folder_name (workspace_id)
    chunks_by_workspace = {}
    for doc in chunked_docs:
        workspace_id = doc.metadata.get("folder_name", "default_workspace")
        if workspace_id not in chunks_by_workspace:
            chunks_by_workspace[workspace_id] = []
        
        chunks_by_workspace[workspace_id].append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })

    # Saving in smaller batches per workspace
    batch_size = 100
    for workspace_id, contents in chunks_by_workspace.items():
        print(f"\n📂 Ingesting into workspace: {workspace_id} ({len(contents)} items)")
        
        for i in range(0, len(contents), batch_size):
            sub_batch = contents[i:i + batch_size]
            print(f"  Ingesting batch {i//batch_size + 1} ({len(sub_batch)} items)...")
            await store_service.save_knowledge_batch(
                workspace_id=workspace_id,
                contents=sub_batch,
                source="file_ingestion"
            )

    print("\n✅ Ingestion Pipeline Completed Successfully!")

if __name__ == "__main__":
    asyncio.run(ingest_pipeline())
