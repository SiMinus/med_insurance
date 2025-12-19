"""
数据加载模块
加载和预处理SQuAD v2数据集用于chunking实验
"""

import json
import os
from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm


class DataLoader:
    """SQuAD数据集加载器"""
    
    def __init__(self, config: Dict):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.subset_size = config['dataset'].get('subset_size')
        self.split = config['dataset']['split']
        self.cache_dir = config['dataset']['cache_dir']
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        加载SQuAD数据集
        
        Returns:
            (documents, qa_pairs): 文档列表和问答对列表
        """
        print(f"正在加载 {self.dataset_name} 数据集 (split: {self.split})...")
        
        # 检查本地缓存
        local_cache_file = os.path.join(self.cache_dir, "processed", f"squad_{self.split}_processed.json")
        if os.path.exists(local_cache_file):
            print(f"从本地缓存加载: {local_cache_file}")
            with open(local_cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            return cached_data['documents'], cached_data['qa_pairs']
        
        # 加载SQuAD v2数据集
        dataset = load_dataset(
            "squad_v2",
            split=self.split,
            cache_dir=os.path.join(self.cache_dir, "raw")
        )
        
        # 打印shuffle前的数据(索引6-10)
        print("\nShuffle前 (索引6-10):")
        for i in range(6, min(11, len(dataset))):
            print(f"  [{i}] {dataset[i]['question'][:60]}...")
        
        # Shuffle数据集(给定随机种子42保证可重复性)
        dataset = dataset.shuffle(seed=42)
        
        # 打印shuffle后的数据(索引6-10)
        print("\nShuffle后 (索引6-10):")
        for i in range(6, min(11, len(dataset))):
            print(f"  [{i}] {dataset[i]['question'][:60]}...")
        print()
        
        # 限制数据集大小
        if self.subset_size and self.subset_size < len(dataset):
            dataset = dataset.select(range(self.subset_size))
        
        print(f"数据集大小: {len(dataset)} 条")
        
        # 处理数据
        documents = []
        qa_pairs = []
        doc_id_map = {}  # 用于去重: context -> doc_id
        
        for idx, item in enumerate(tqdm(dataset, desc="处理数据")):
            context = item['context']
            question = item['question']
            answers = item['answers']
            
            # 添加文档(去重)
            if context not in doc_id_map:
                doc_id = len(documents)
                doc_id_map[context] = doc_id
                documents.append({
                    'id': doc_id,
                    'text': context,
                    'title': item.get('title', f'Document {doc_id}')
                })
            else:
                doc_id = doc_id_map[context]
            
            # 添加问答对
            qa_pairs.append({
                'id': idx,
                'question': question,
                'answers': answers['text'] if answers['text'] else [],  # SQuAD v2可能没有答案
                'doc_id': doc_id,
                'context': context
            })
        
        print(f"处理完成: {len(documents)} 个文档, {len(qa_pairs)} 个问答对")
        
        # 保存到本地缓存
        os.makedirs(os.path.dirname(local_cache_file), exist_ok=True)
        with open(local_cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': documents,
                'qa_pairs': qa_pairs
            }, f, ensure_ascii=False, indent=2)
        print(f"已缓存到本地: {local_cache_file}")
        
        return documents, qa_pairs
    



def load_squad_data(config: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    便捷函数: 根据配置加载SQuAD数据
    
    Args:
        config: 配置字典
        
    Returns:
        (documents, train_queries, test_queries)
    """
    loader = DataLoader(config)
    documents, qa_pairs = loader.load_data()
    
    # 划分训练集和测试集
    train_queries, test_queries = loader.split_data(
        qa_pairs,
        test_size=config['evaluation']['test_size']
    )
    
    return documents, train_queries, test_queries


if __name__ == "__main__":
    """主函数 - 用于直接运行数据加载"""
    import yaml
    
    # 加载配置
    with open('../config/chunking_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 测试数据加载
    documents, train_queries, test_queries = load_squad_data(config)
    
    print("\n=== 数据加载测试 ===")
    print(f"Documents: {len(documents)}")
    print(f"Train Queries: {len(train_queries)}")
    print(f"Test Queries: {len(test_queries)}")
    
    print("\n示例Document:")
    print(documents[0])
    
    print("\n示例Query:")
    print(test_queries[0])
