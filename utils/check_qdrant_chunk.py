from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path

def check_chunk(chunk_id):
    # Path to the local Qdrant storage
    QDRANT_PATH = str(Path(__file__).parent.parent / "qdrant_storage")
    COLLECTION_NAME = "medical_insurance_chunks"
    
    print(f"Connecting to Qdrant at: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)
    
    print(f"Searching for chunk_id: {chunk_id} in collection: {COLLECTION_NAME}")
    
    try:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.chunk_id",
                        match=models.MatchValue(value=chunk_id)
                    )
                ]
            ),
            limit=1
        )
        
        points, _ = response
        
        if points:
            print(f"\n✅ 成功找到 Chunk: {chunk_id}")
            print("-" * 30)
            payload = points[0].payload
            print(f"Page: {payload['metadata']['page']}")
            print(f"Filename: {payload['metadata']['filename']}")
            print(f"Text Content:\n{payload['text']}")
            print("-" * 30)
            return True
        else:
            print(f"\n❌ 未找到 Chunk: {chunk_id}")
            return False
            
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return False

if __name__ == "__main__":
    check_chunk("chunk_96")
