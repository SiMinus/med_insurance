"""
将chunks上传到本地Qdrant向量数据库
包含元数据：文件名、页码
"""
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 导入所有chunks
from qdrant_chunks import (
    chunk_one, chunk_two, chunk_1, chunk_2, chunk_3, chunk_4, chunk_5,
    chunk_6, chunk_7, chunk_8, chunk_9, chunk_10, chunk_11, chunk_12,
    chunk_13, chunk_14, chunk_15, chunk_16, chunk_17, chunk_18, chunk_19
)


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """创建集合"""
    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        client.delete_collection(collection_name=collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def prepare_chunks_with_metadata():
    """准备chunks数据及元数据"""
    filename = "小红书医保待遇（最终版）.pdf"
    
    chunks_data = [
        # 第一页
        {"text": chunk_one.strip(), "page": 1, "chunk_id": "chunk_one"},
        {"text": chunk_two.strip(), "page": 1, "chunk_id": "chunk_two"},
        {"text": chunk_1.strip(), "page": 1, "chunk_id": "chunk_1"},
        {"text": chunk_2.strip(), "page": 1, "chunk_id": "chunk_2"},
        {"text": chunk_3.strip(), "page": 1, "chunk_id": "chunk_3"},
        {"text": chunk_4.strip(), "page": 1, "chunk_id": "chunk_4"},
        {"text": chunk_5.strip(), "page": 1, "chunk_id": "chunk_5"},
        
        # 第二页
        {"text": chunk_6.strip(), "page": 2, "chunk_id": "chunk_6"},
        {"text": chunk_7.strip(), "page": 2, "chunk_id": "chunk_7"},
        {"text": chunk_8.strip(), "page": 2, "chunk_id": "chunk_8"},
        {"text": chunk_9.strip(), "page": 2, "chunk_id": "chunk_9"},
        {"text": chunk_10.strip(), "page": 2, "chunk_id": "chunk_10"},
        {"text": chunk_11.strip(), "page": 2, "chunk_id": "chunk_11"},
        {"text": chunk_12.strip(), "page": 2, "chunk_id": "chunk_12"},
        
        # 第三页
        {"text": chunk_13.strip(), "page": 3, "chunk_id": "chunk_13"},
        {"text": chunk_14.strip(), "page": 3, "chunk_id": "chunk_14"},
        {"text": chunk_15.strip(), "page": 3, "chunk_id": "chunk_15"},
        {"text": chunk_16.strip(), "page": 3, "chunk_id": "chunk_16"},
        {"text": chunk_17.strip(), "page": 3, "chunk_id": "chunk_17"},
        {"text": chunk_18.strip(), "page": 3, "chunk_id": "chunk_18"},
        {"text": chunk_19.strip(), "page": 3, "chunk_id": "chunk_19"},
    ]
    
    for chunk in chunks_data:
        chunk["filename"] = filename
    
    return chunks_data


def upload_chunks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks_data: list,
    embedding_model: SentenceTransformer
):
    """将chunks上传到Qdrant"""
    texts = [chunk["text"] for chunk in chunks_data]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    points = []
    for chunk, embedding in zip(chunks_data, embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": chunk["text"],
                "metadata": {
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                }
            }
        )
        points.append(point)
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"已上传 {len(points)} 个chunks")


def main():
    """主函数"""
    # 配置
    COLLECTION_NAME = "medical_insurance_chunks"
    QDRANT_PATH = "./qdrant_storage"
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    print(f"集合名称: {COLLECTION_NAME}")
    print(f"本地存储: {QDRANT_PATH}")
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    
    # 初始化客户端
    client = QdrantClient(path=QDRANT_PATH)
    
    # 加载模型
    print(f"\n加载嵌入模型...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # 获取向量维度
    sample_embedding = embedding_model.encode(["测试"])[0]
    vector_size = len(sample_embedding)
    
    # 创建集合
    print(f"\n创建集合...")
    create_collection(client, COLLECTION_NAME, vector_size)
    
    # 准备数据
    print(f"\n准备数据...")
    chunks_data = prepare_chunks_with_metadata()
    print(f"已准备 {len(chunks_data)} 个chunks")
    
    # 上传数据
    print(f"\n上传数据...")
    upload_chunks_to_qdrant(client, COLLECTION_NAME, chunks_data, embedding_model)
    
    print(f"\n完成！")


if __name__ == "__main__":
    main()
