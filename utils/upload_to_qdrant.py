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
    # chunk_one, chunk_two, chunk_1, chunk_2, chunk_3, chunk_4, chunk_5,
    # chunk_6, chunk_7, chunk_8, chunk_9, chunk_10, chunk_11, chunk_12,
    # chunk_13, chunk_14, chunk_15, chunk_16, chunk_17, chunk_18, chunk_19,
    chunk_20, chunk_21, chunk_22, chunk_23, chunk_24, chunk_25, chunk_26,
    chunk_27, chunk_28, chunk_29, chunk_30, chunk_31, chunk_32, chunk_33,
    chunk_34, chunk_35, chunk_36, chunk_37, chunk_38, chunk_39, chunk_40,
    chunk_41, chunk_42, chunk_43, chunk_44, chunk_45, chunk_46, chunk_47,
    chunk_48, chunk_49, chunk_50, chunk_51, chunk_52, chunk_53, chunk_54,
    chunk_55, chunk_56, chunk_57, chunk_58, chunk_59, chunk_60, chunk_61,
    chunk_62, chunk_63, chunk_64, chunk_65, chunk_66, chunk_67, chunk_68,
    chunk_69, chunk_70, chunk_71, chunk_72, chunk_73, chunk_74, chunk_75,
    chunk_76, chunk_77, chunk_78, chunk_79, chunk_80, chunk_81, chunk_82,
    chunk_83, chunk_84, chunk_85, chunk_86, chunk_87, chunk_88, chunk_89,
    chunk_90, chunk_91, chunk_92, chunk_93, chunk_94, chunk_95, chunk_96
)


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """创建集合（如果不存在）"""
    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"集合 '{collection_name}' 已存在，跳过创建")
        return
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"集合 '{collection_name}' 创建成功")


def prepare_chunks_with_metadata():
    """准备chunks数据及元数据"""
    filename = "小红书医保待遇（最终版）.pdf"
    
    chunks_data = [
        # # 第一页
        # {"text": chunk_one.strip(), "page": 1, "chunk_id": "chunk_one"},
        # {"text": chunk_two.strip(), "page": 1, "chunk_id": "chunk_two"},
        # {"text": chunk_1.strip(), "page": 1, "chunk_id": "chunk_1"},
        # {"text": chunk_2.strip(), "page": 1, "chunk_id": "chunk_2"},
        # {"text": chunk_3.strip(), "page": 1, "chunk_id": "chunk_3"},
        # {"text": chunk_4.strip(), "page": 1, "chunk_id": "chunk_4"},
        # {"text": chunk_5.strip(), "page": 1, "chunk_id": "chunk_5"},
        
        # # 第二页
        # {"text": chunk_6.strip(), "page": 2, "chunk_id": "chunk_6"},
        # {"text": chunk_7.strip(), "page": 2, "chunk_id": "chunk_7"},
        # {"text": chunk_8.strip(), "page": 2, "chunk_id": "chunk_8"},
        # {"text": chunk_9.strip(), "page": 2, "chunk_id": "chunk_9"},
        # {"text": chunk_10.strip(), "page": 2, "chunk_id": "chunk_10"},
        # {"text": chunk_11.strip(), "page": 2, "chunk_id": "chunk_11"},
        # {"text": chunk_12.strip(), "page": 2, "chunk_id": "chunk_12"},
        
        # # 第三页
        # {"text": chunk_13.strip(), "page": 3, "chunk_id": "chunk_13"},
        # {"text": chunk_14.strip(), "page": 3, "chunk_id": "chunk_14"},
        # {"text": chunk_15.strip(), "page": 3, "chunk_id": "chunk_15"},
        # {"text": chunk_16.strip(), "page": 3, "chunk_id": "chunk_16"},
        # {"text": chunk_17.strip(), "page": 3, "chunk_id": "chunk_17"},
        # {"text": chunk_18.strip(), "page": 3, "chunk_id": "chunk_18"},
        # {"text": chunk_19.strip(), "page": 3, "chunk_id": "chunk_19"},

        # 第4-5页
        {"text": chunk_20.strip(), "page": 4, "chunk_id": "chunk_20"},
        {"text": chunk_21.strip(), "page": 4, "chunk_id": "chunk_21"},
        {"text": chunk_22.strip(), "page": 4, "chunk_id": "chunk_22"},
        {"text": chunk_23.strip(), "page": 4, "chunk_id": "chunk_23"},
        {"text": chunk_24.strip(), "page": 4, "chunk_id": "chunk_24"},

        # 第6-7页
        {"text": chunk_25.strip(), "page": 6, "chunk_id": "chunk_25"},
        {"text": chunk_26.strip(), "page": 6, "chunk_id": "chunk_26"},
        {"text": chunk_27.strip(), "page": 6, "chunk_id": "chunk_27"},
        {"text": chunk_28.strip(), "page": 6, "chunk_id": "chunk_28"},
        {"text": chunk_29.strip(), "page": 6, "chunk_id": "chunk_29"},
        {"text": chunk_30.strip(), "page": 6, "chunk_id": "chunk_30"},
        {"text": chunk_31.strip(), "page": 6, "chunk_id": "chunk_31"},

        # 第8-9页
        {"text": chunk_32.strip(), "page": 8, "chunk_id": "chunk_32"},
        {"text": chunk_33.strip(), "page": 8, "chunk_id": "chunk_33"},
        {"text": chunk_34.strip(), "page": 8, "chunk_id": "chunk_34"},
        {"text": chunk_35.strip(), "page": 8, "chunk_id": "chunk_35"},
        {"text": chunk_36.strip(), "page": 8, "chunk_id": "chunk_36"},
        {"text": chunk_37.strip(), "page": 8, "chunk_id": "chunk_37"},
        {"text": chunk_38.strip(), "page": 8, "chunk_id": "chunk_38"},
        {"text": chunk_39.strip(), "page": 8, "chunk_id": "chunk_39"},
        {"text": chunk_40.strip(), "page": 8, "chunk_id": "chunk_40"},
        {"text": chunk_41.strip(), "page": 8, "chunk_id": "chunk_41"},
        {"text": chunk_42.strip(), "page": 8, "chunk_id": "chunk_42"},
        {"text": chunk_43.strip(), "page": 8, "chunk_id": "chunk_43"},
        {"text": chunk_44.strip(), "page": 8, "chunk_id": "chunk_44"},

        # 第10-11页
        {"text": chunk_45.strip(), "page": 10, "chunk_id": "chunk_45"},
        {"text": chunk_46.strip(), "page": 10, "chunk_id": "chunk_46"},
        {"text": chunk_47.strip(), "page": 10, "chunk_id": "chunk_47"},
        {"text": chunk_48.strip(), "page": 10, "chunk_id": "chunk_48"},
        {"text": chunk_49.strip(), "page": 10, "chunk_id": "chunk_49"},

        # 第12-13页
        {"text": chunk_50.strip(), "page": 12, "chunk_id": "chunk_50"},
        {"text": chunk_51.strip(), "page": 12, "chunk_id": "chunk_51"},
        {"text": chunk_52.strip(), "page": 12, "chunk_id": "chunk_52"},
        {"text": chunk_53.strip(), "page": 12, "chunk_id": "chunk_53"},
        {"text": chunk_54.strip(), "page": 12, "chunk_id": "chunk_54"},

        # 第14-15页
        {"text": chunk_55.strip(), "page": 14, "chunk_id": "chunk_55"},
        {"text": chunk_56.strip(), "page": 14, "chunk_id": "chunk_56"},
        {"text": chunk_57.strip(), "page": 14, "chunk_id": "chunk_57"},
        {"text": chunk_58.strip(), "page": 14, "chunk_id": "chunk_58"},
        {"text": chunk_59.strip(), "page": 14, "chunk_id": "chunk_59"},
        {"text": chunk_60.strip(), "page": 14, "chunk_id": "chunk_60"},
        {"text": chunk_61.strip(), "page": 14, "chunk_id": "chunk_61"},

        # 第16-17页
        {"text": chunk_62.strip(), "page": 16, "chunk_id": "chunk_62"},
        {"text": chunk_63.strip(), "page": 16, "chunk_id": "chunk_63"},
        {"text": chunk_64.strip(), "page": 16, "chunk_id": "chunk_64"},
        {"text": chunk_65.strip(), "page": 16, "chunk_id": "chunk_65"},
        {"text": chunk_66.strip(), "page": 16, "chunk_id": "chunk_66"},
        {"text": chunk_67.strip(), "page": 16, "chunk_id": "chunk_67"},

        # 第18-19页
        {"text": chunk_68.strip(), "page": 18, "chunk_id": "chunk_68"},
        {"text": chunk_69.strip(), "page": 18, "chunk_id": "chunk_69"},
        {"text": chunk_70.strip(), "page": 18, "chunk_id": "chunk_70"},
        {"text": chunk_71.strip(), "page": 18, "chunk_id": "chunk_71"},
        {"text": chunk_72.strip(), "page": 18, "chunk_id": "chunk_72"},

        # 第20-21页
        {"text": chunk_73.strip(), "page": 20, "chunk_id": "chunk_73"},
        {"text": chunk_74.strip(), "page": 20, "chunk_id": "chunk_74"},
        {"text": chunk_75.strip(), "page": 20, "chunk_id": "chunk_75"},
        {"text": chunk_76.strip(), "page": 20, "chunk_id": "chunk_76"},
        {"text": chunk_77.strip(), "page": 20, "chunk_id": "chunk_77"},

        # 第22-23页
        {"text": chunk_78.strip(), "page": 22, "chunk_id": "chunk_78"},
        {"text": chunk_79.strip(), "page": 22, "chunk_id": "chunk_79"},
        {"text": chunk_80.strip(), "page": 22, "chunk_id": "chunk_80"},
        {"text": chunk_81.strip(), "page": 22, "chunk_id": "chunk_81"},
        {"text": chunk_82.strip(), "page": 22, "chunk_id": "chunk_82"},

        # 第24-25页
        {"text": chunk_83.strip(), "page": 24, "chunk_id": "chunk_83"},
        {"text": chunk_84.strip(), "page": 24, "chunk_id": "chunk_84"},
        {"text": chunk_85.strip(), "page": 24, "chunk_id": "chunk_85"},
        {"text": chunk_86.strip(), "page": 24, "chunk_id": "chunk_86"},

        # 第26-27页
        {"text": chunk_87.strip(), "page": 26, "chunk_id": "chunk_87"},
        {"text": chunk_88.strip(), "page": 26, "chunk_id": "chunk_88"},
        {"text": chunk_89.strip(), "page": 26, "chunk_id": "chunk_89"},
        {"text": chunk_90.strip(), "page": 26, "chunk_id": "chunk_90"},
        {"text": chunk_91.strip(), "page": 26, "chunk_id": "chunk_91"},

        # 第28-29页
        {"text": chunk_92.strip(), "page": 28, "chunk_id": "chunk_92"},
        {"text": chunk_93.strip(), "page": 28, "chunk_id": "chunk_93"},
        {"text": chunk_94.strip(), "page": 28, "chunk_id": "chunk_94"},
        {"text": chunk_95.strip(), "page": 28, "chunk_id": "chunk_95"},

        # 第30页
        {"text": chunk_96.strip(), "page": 30, "chunk_id": "chunk_96"},
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
    # 使用项目根目录的 qdrant_storage（相对于脚本位置的上一级目录）
    QDRANT_PATH = str(Path(__file__).parent.parent / "qdrant_storage")
    EMBEDDING_MODEL = "BAAI/bge-m3"
    
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
