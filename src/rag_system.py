"""
RAG系统实现
包含向量数据库、检索和生成组件
"""

import time
from typing import List, Dict, Tuple
import numpy as np
try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
import chromadb
from chromadb.config import Settings
from src.embeddings import EmbeddingModel


class VectorDatabase:
    """向量数据库 - 支持FAISS或ChromaDB"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        初始化向量数据库
        
        Args:
            embedding_model: Embedding模型
        """
        self.embedding_model = embedding_model
        self.dimension = embedding_model.dimension
        self.use_faiss = USE_FAISS
        self.documents = []
        self.id_to_index = {}  # ChromaDB用: ID字符串 -> 文档索引映射
        
        if self.use_faiss:
            self.index = None
            print("使用 FAISS 作为向量数据库")
        else:
            # 使用ChromaDB
            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            self.collection_name = f"rag_collection_{embedding_model.model_name}"
            print("使用 ChromaDB 作为向量数据库")
        
    def build_index(self, documents: List[Dict]):
        """
        构建向量索引
        
        Args:
            documents: 文档列表 [{'id': ..., 'text': ..., 'title': ...}, ...]
        """
        print(f"正在构建向量索引 (模型: {self.embedding_model.model_name})...")
        
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        # 编码文本
        start_time = time.time()
        # 使用更大的batch_size提升速度 (默认32 -> 128)
        embeddings = self.embedding_model.encode(texts, batch_size=128)
        encoding_time = time.time() - start_time
        
        if self.use_faiss:
            # 使用FAISS (embeddings已是numpy float32)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)  # 直接添加,无需astype转换
        else:
            # 使用ChromaDB
            # 重置集合
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 添加文档
            ids = [str(doc['id']) for doc in documents]
            metadatas = [{'title': doc.get('title', ''), 'original_id': doc['id']} for doc in documents]
            
            # 创建ID到索引的映射(用于检索时快速查找)
            self.id_to_index = {str(doc['id']): idx for idx, doc in enumerate(documents)}
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"索引构建完成: {len(documents)} 个文档, 耗时 {encoding_time:.2f}s")
        
        return encoding_time
    
    def search(self, query: str, top_k: int = 3) -> Tuple[List[Dict], float]:
        """
        检索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            (retrieved_docs, search_time): 检索到的文档列表和检索时间(不含编码时间)
        """
        # 编码查询(不计入search_time)
        query_embedding = self.embedding_model.encode(query)
        
        # FAISS优化: 提前转换避免在计时内重复传输
        if self.use_faiss:
            query_embedding = query_embedding.astype('float32')
        
        # 从这里开始计时(纯检索时间)
        start_time = time.time()
        
        if self.use_faiss:
            # 使用FAISS搜索
            if self.index is None:
                raise RuntimeError("请先构建索引!")
            
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1),
                top_k
            )
            
            search_time = time.time() - start_time
            
            # 整理结果
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(distances[0][i])
                    doc['rank'] = i + 1
                    retrieved_docs.append(doc)
        else:
            # 使用ChromaDB搜索
            # 确保query_embedding是一维列表
            if isinstance(query_embedding, np.ndarray):
                if query_embedding.ndim > 1:
                    query_embedding = query_embedding.flatten()
                query_embedding_list = query_embedding.tolist()
            else:
                query_embedding_list = query_embedding
            
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k
            )
            
            search_time = time.time() - start_time
            
            # 整理结果
            retrieved_docs = []
            for i in range(len(results['ids'][0])):
                doc_id_str = results['ids'][0][i]  # ChromaDB返回的是字符串ID
                # 通过ID映射找到文档索引
                if doc_id_str in self.id_to_index:
                    doc_idx = self.id_to_index[doc_id_str]
                    doc = self.documents[doc_idx].copy()
                    doc['score'] = results['distances'][0][i] if results['distances'] else 0.0
                    doc['rank'] = i + 1
                    retrieved_docs.append(doc)
        
        return retrieved_docs, search_time


class RAGSystem:
    """RAG系统"""
    
    def __init__(self, embedding_model: EmbeddingModel, config: Dict):
        """
        初始化RAG系统
        
        Args:
            embedding_model: Embedding模型
            config: 配置字典
        """
        self.embedding_model = embedding_model
        self.config = config
        self.vector_db = VectorDatabase(embedding_model)
        self.top_k = config['rag']['top_k']
        
    def index_documents(self, documents: List[Dict]) -> float:
        """
        索引文档
        
        Args:
            documents: 文档列表
            
        Returns:
            索引构建时间
        """
        return self.vector_db.build_index(documents)
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[Dict], float]:
        """
        检索相关文档
        
        Args:
            query: 查询问题
            top_k: 返回的文档数量(如果为None则使用配置中的值)
            
        Returns:
            (retrieved_docs, retrieval_time): 检索结果和检索时间
        """
        if top_k is None:
            top_k = self.top_k
        
        return self.vector_db.search(query, top_k)
    
    def answer_question(self, question: str, top_k: int = None) -> Dict:
        """
        回答问题(检索+生成)
        
        Args:
            question: 问题
            top_k: 检索的文档数量
            
        Returns:
            结果字典，包含检索到的文档和答案
        """
        # 检索相关文档
        retrieved_docs, retrieval_time = self.retrieve(question, top_k)
        
        # 简单的答案生成(这里只是示例,实际应该用LLM生成)
        # 在真实场景中,这里会调用GPT等模型
        if retrieved_docs:
            # 取最相关的文档作为答案来源
            best_doc = retrieved_docs[0]
            # 在实际应用中,这里应该用LLM基于context生成答案
            answer = f"根据检索到的文档: {best_doc['text'][:200]}..."
        else:
            answer = "未找到相关信息"
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieval_time': retrieval_time,
            'num_docs_retrieved': len(retrieved_docs)
        }
    
    def batch_retrieve(self, questions: List[str]) -> List[Dict]:
        """
        批量检索
        
        Args:
            questions: 问题列表
            
        Returns:
            结果列表
        """
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        
        return results


class TextChunker:
    """文本分块工具 - 支持三种分块策略"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Fixed Size: 固定字符数分块
        
        Args:
            text: 原始文本
            chunk_size: 每块的字符数
            overlap: 重叠字符数
            
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    @staticmethod
    def chunk_text_by_sentence(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Sentence-based: 按句子边界分块
        
        Args:
            text: 原始文本
            chunk_size: 每块的最大字符数
            overlap: 重叠字符数
            
        Returns:
            文本块列表
        """
        import re
        # 按句子分割（支持中英文标点）
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ''
        
        for sentence in sentences:
            # 如果添加当前句子不超过chunk_size，则合并
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + ' '
            else:
                # 否则保存当前chunk，开始新chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 处理overlap：让相邻chunk有重叠部分
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    # 从前一个chunk末尾取overlap长度的文本
                    prev_overlap = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                    overlapped_chunks.append(prev_overlap + ' ' + chunks[i])
            return overlapped_chunks
        
        return chunks
    
    @staticmethod
    def chunk_text_semantic(text: str, embedding_model, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Semantic: 基于语义相似度分块
        
        Args:
            text: 原始文本
            embedding_model: 用于计算语义相似度的embedding模型
            chunk_size: 每块的最大字符数
            overlap: 重叠字符数
            
        Returns:
            文本块列表
        """
        import re
        # 按句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        if len(sentences) == 1:
            return sentences
        
        chunks = []
        current_chunk = sentences[0]
        
        for i in range(1, len(sentences)):
            # 计算当前chunk和下一个句子的语义相似度
            try:
                emb_chunk = embedding_model.encode([current_chunk])[0]
                emb_next = embedding_model.encode([sentences[i]])[0]
                
                # 计算余弦相似度
                similarity = float(np.dot(emb_chunk, emb_next) / 
                                 (np.linalg.norm(emb_chunk) * np.linalg.norm(emb_next) + 1e-8))
                
                # 如果相似度高且长度未超限，则合并
                if similarity > 0.7 and len(current_chunk) + len(sentences[i]) <= chunk_size:
                    current_chunk += ' ' + sentences[i]
                else:
                    # 否则保存当前chunk，开始新chunk
                    chunks.append(current_chunk.strip())
                    current_chunk = sentences[i]
            except Exception as e:
                print(f"语义分块出错: {e}，使用固定分块")
                # 出错时使用固定分块
                if len(current_chunk) + len(sentences[i]) <= chunk_size:
                    current_chunk += ' ' + sentences[i]
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentences[i]
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 处理overlap
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    prev_overlap = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                    overlapped_chunks.append(prev_overlap + ' ' + chunks[i])
            return overlapped_chunks
        
        return chunks
    
    @staticmethod
    def chunk_text_recursive(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Recursive: 递归分块，按分隔符层级递归分割
        优先按段落 → 句子 → 单词分割，保持语义完整性
        
        Args:
            text: 原始文本
            chunk_size: 每块的最大字符数
            overlap: 重叠字符数
            
        Returns:
            文本块列表
        """
        # 定义分隔符层级（从粗到细）
        separators = [
            "\n\n",      # 段落
            "\n",        # 行
            ". ",        # 句子（英文）
            "。",        # 句子（中文）
            "! ",        # 感叹句
            "？",        # 问句
            " ",         # 单词
            ""           # 字符
        ]
        
        def _recursive_split(text: str, separators: List[str]) -> List[str]:
            """递归分割文本"""
            # 如果文本已经小于chunk_size，直接返回
            if len(text) <= chunk_size:
                return [text] if text else []
            
            # 如果没有分隔符了，强制切分
            if not separators:
                chunks = []
                for i in range(0, len(text), chunk_size - overlap):
                    chunks.append(text[i:i + chunk_size])
                return chunks
            
            # 使用当前层级的分隔符分割
            separator = separators[0]
            remaining_separators = separators[1:]
            
            if separator == "":
                # 最后一级：按字符切分
                chunks = []
                for i in range(0, len(text), chunk_size - overlap):
                    chunks.append(text[i:i + chunk_size])
                return chunks
            
            # 按分隔符分割
            splits = text.split(separator)
            
            # 重新组合，保留分隔符
            chunks = []
            current_chunk = ""
            
            for i, split in enumerate(splits):
                # 恢复分隔符（除了最后一个）
                piece = split + (separator if i < len(splits) - 1 else "")
                
                # 如果单个piece就超过chunk_size，递归处理
                if len(piece) > chunk_size:
                    # 先保存当前chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    # 递归分割大piece
                    sub_chunks = _recursive_split(piece, remaining_separators)
                    chunks.extend(sub_chunks)
                # 如果加上这个piece不超过chunk_size，合并
                elif len(current_chunk) + len(piece) <= chunk_size:
                    current_chunk += piece
                # 否则保存当前chunk，开始新chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = piece
            
            # 添加最后一个chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        
        # 执行递归分割
        chunks = _recursive_split(text, separators)
        
        # 处理overlap
        if overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    # 从前一个chunk末尾取overlap长度
                    prev_overlap = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                    overlapped_chunks.append(prev_overlap + chunks[i])
            return overlapped_chunks
        
        return chunks
    
    @staticmethod
    def chunk_documents(documents: List[Dict], chunk_size: int = 200, overlap: int = 50, 
                       strategy: str = "fixed", embedding_model=None) -> List[Dict]:
        """
        将文档列表分块 - 支持四种分块策略
        
        Args:
            documents: 文档列表
            chunk_size: 每块的字符数
            overlap: 重叠字符数
            strategy: 分块策略 ('fixed', 'sentence', 'semantic', 'recursive')
            embedding_model: embedding模型（semantic策略必需）
            
        Returns:
            分块后的文档列表
        """
        chunked_docs = []
        
        for doc in documents:
            # 根据策略选择分块方法
            if strategy == "fixed":
                chunks = TextChunker.chunk_text(doc['text'], chunk_size, overlap)
            elif strategy == "sentence":
                chunks = TextChunker.chunk_text_by_sentence(doc['text'], chunk_size, overlap)
            elif strategy == "semantic":
                if embedding_model is None:
                    raise ValueError("Semantic chunking需要提供embedding_model参数")
                chunks = TextChunker.chunk_text_semantic(doc['text'], embedding_model, chunk_size, overlap)
            elif strategy == "recursive":
                chunks = TextChunker.chunk_text_recursive(doc['text'], chunk_size, overlap)
            else:
                raise ValueError(f"未知的分块策略: {strategy}。支持的策略: 'fixed', 'sentence', 'semantic', 'recursive'")
            
            # 为每个chunk创建文档
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    'id': f"{doc['id']}_chunk_{i}",
                    'text': chunk,
                    'title': doc.get('title', ''),
                    'original_doc_id': doc['id'],
                    'chunk_index': i
                })
        
        return chunked_docs


if __name__ == "__main__":
    # 测试代码
    from src.embeddings import SentenceTransformerModel
    
    # 创建测试数据
    test_docs = [
        {'id': 0, 'text': 'Python is a programming language.', 'title': 'Doc1'},
        {'id': 1, 'text': 'Machine learning is a subset of AI.', 'title': 'Doc2'},
        {'id': 2, 'text': 'RAG combines retrieval and generation.', 'title': 'Doc3'}
    ]
    
    # 创建模型和RAG系统
    model = SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2", "test")
    config = {'rag': {'top_k': 2}}
    
    rag = RAGSystem(model, config)
    rag.index_documents(test_docs)
    
    # 测试检索
    result = rag.answer_question("What is RAG?")
    print(f"Question: {result['question']}")
    print(f"Retrieved {result['num_docs_retrieved']} documents in {result['retrieval_time']:.4f}s")
