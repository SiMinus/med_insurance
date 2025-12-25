"""
向量模型封装模块
支持多种Embedding模型
"""

import os
import time
from typing import List, Union
from abc import ABC, abstractmethod
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel(ABC):
    """Embedding模型基类"""
    
    def __init__(self, model_id: str, model_name: str):
        self.model_id = model_id
        self.model_name = model_name
        self.dimension = None
        
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            
        Returns:
            向量数组
        """
        pass
    
    def encode_with_timing(self, texts: Union[str, List[str]]) -> tuple:
        """
        编码并记录时间
        
        Returns:
            (embeddings, elapsed_time)
        """
        start_time = time.time()
        embeddings = self.encode(texts)
        elapsed_time = time.time() - start_time
        return embeddings, elapsed_time


class SentenceTransformerModel(EmbeddingModel):
    """Sentence-Transformers模型"""
    
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        print(f"正在加载模型: {model_id}")
        
        # 自动检测设备: MPS (Mac) > CUDA (NVIDIA) > CPU
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
            device_name = 'Apple Silicon (MPS)'
        elif torch.cuda.is_available():
            device = 'cuda'
            device_name = f'NVIDIA {torch.cuda.get_device_name(0)}'
        else:
            device = 'cpu'
            device_name = 'CPU'
        
        self.model = SentenceTransformer(model_id, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"模型加载完成, 维度: {self.dimension}, 设备: {device_name}")
        
        if device in ['mps', 'cuda']:
            print(f"  ✓ 使用GPU加速")
        
    def encode(self, texts: Union[str, List[str]], batch_size: int = 128, show_progress_bar: bool = None) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 自动决定是否显示进度条(大于1000条才显示)
        if show_progress_bar is None:
            show_progress_bar = len(texts) > 10
        
        # 优化: 返回Tensor避免提前MPS→CPU传输，由调用方决定何时转换
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=False,  # 保持Tensor格式
            convert_to_tensor=True,   # 返回单个Tensor而非list
            normalize_embeddings=False
        )
        
        # 优化: 确保float32再传输到CPU(顺序很重要!)
        # 先.to()在MPS上完成(无开销),再.cpu()传输(更快)
        return embeddings.to(torch.float32).cpu().numpy()


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI Embedding模型"""
    
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model_id = model_id
            
            # 根据模型设置维度
            if 'small' in model_id:
                self.dimension = 1536
            elif 'large' in model_id:
                self.dimension = 3072
            else:
                self.dimension = 1536
                
            print(f"OpenAI模型初始化完成: {model_id}, 维度: {self.dimension}")
            
        except ImportError:
            raise ImportError("请安装openai包: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI模型初始化失败: {e}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        # OpenAI API批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_id
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)


class EmbeddingModelFactory:
    """Embedding模型工厂"""
    
    @staticmethod
    def create_model(config: dict) -> EmbeddingModel:
        """
        根据配置创建模型
        
        Args:
            config: 模型配置字典
            
        Returns:
            EmbeddingModel实例
        """
        model_type = config['type']
        model_id = config['model_id']
        model_name = config['name']
        
        if model_type == 'sentence-transformers':
            return SentenceTransformerModel(model_id, model_name)
        elif model_type == 'openai':
            return OpenAIEmbeddingModel(model_id, model_name)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def create_all_models(model_configs: List[dict]) -> dict:
        """
        创建所有配置的模型
        
        Args:
            model_configs: 模型配置列表
            
        Returns:
            {model_name: EmbeddingModel} 字典
        """
        models = {}
        
        for config in model_configs:
            try:
                model = EmbeddingModelFactory.create_model(config)
                models[config['name']] = model
                print(f"✓ 模型 '{config['name']}' 加载成功")
            except Exception as e:
                print(f"✗ 模型 '{config['name']}' 加载失败: {e}")
        
        return models


if __name__ == "__main__":
    # 测试代码
    model = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2",
        "test"
    )
    
    test_texts = ["This is a test sentence.", "Another test sentence."]
    embeddings, elapsed = model.encode_with_timing(test_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Time elapsed: {elapsed:.4f}s")
