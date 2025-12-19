"""
评估模块
使用Ragas库评估RAG系统的性能指标
"""

import time
import os
from typing import List, Dict
import numpy as np
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import logging

# 禁用tokenizers警告和ragas日志
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger('ragas').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)


class RAGEvaluator:
    """RAG系统评估器 - 基于Ragas"""
    
    def __init__(self, config: Dict):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.metrics = config['evaluation']['metrics']
        self.output_dir = None  # 用于保存评估数据集
        self.chunk_size = None  # 当前实验的chunk_size
        self.overlap = None  # 当前实验的overlap
        
    def set_output_dir(self, output_dir: str):
        """设置输出目录"""
        self.output_dir = output_dir
    
    def set_chunk_config(self, chunk_size: int, overlap: int):
        """设置chunk配置"""
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def evaluate_retrieval(self, rag_system, test_qa_pairs: List[Dict], chunked_docs: List[Dict] = None) -> Dict:
        """
        评估检索性能 - 使用Ragas指标
        
        Args:
            rag_system: RAG系统实例
            test_qa_pairs: 测试问答对列表 [{'question': ..., 'answers': [...], 'doc_id': ..., 'context': ...}]
            chunked_docs: 分块后的文档列表(可选,用于统计)
            
        Returns:
            评估结果字典
        """
        print(f"\n开始评估 (模型: {rag_system.embedding_model.model_name})...")
        
        results = {
            'model_name': rag_system.embedding_model.model_name,
            'model_id': rag_system.embedding_model.model_id,
            'metrics': {}
        }
        
        # 1. 收集检索结果和检索时间
        questions = []
        ground_truths = []
        contexts_list = []
        retrieval_times = []
        
        for qa in test_qa_pairs:
            question = qa['question']
            answers = qa.get('answers', [])
            
            # 跳过无答案的问题(SQuAD v2)
            if not answers:
                continue
            
            # 执行检索
            retrieved_docs, retrieval_time = rag_system.retrieve(question)
            retrieval_times.append(retrieval_time)
            
            # 准备Ragas评估数据
            questions.append(question)
            # Ragas期望ground_truth是字符串,不是列表,将所有答案用' or '连接
            ground_truths.append(" or ".join(answers) if answers else "")
            contexts_list.append([doc['text'] for doc in retrieved_docs])  # 检索到的contexts
        
        print(f"  有效问题数: {len(questions)}")
        
        # 2. 计算检索时间指标
        if retrieval_times:
            results['metrics']['avg_retrieval_time'] = float(np.mean(retrieval_times))
            results['metrics']['total_retrieval_time'] = float(np.sum(retrieval_times))
        
        # 3. 使用Ragas评估context precision和context recall
        if len(questions) > 0:
            try:
                # 构建Ragas数据集
                eval_dataset = Dataset.from_dict({
                    'question': questions,
                    'ground_truth': ground_truths,
                    'contexts': contexts_list
                })
                
                # 先执行Ragas评估获取指标值（移到保存CSV之前）
                ragas_metrics = []
                if 'context_precision' in self.metrics:
                    ragas_metrics.append(context_precision)
                # if 'context_recall' in self.metrics:
                #     ragas_metrics.append(context_recall)
                
                # 用于保存到CSV的列表
                precision_list = []
                # recall_list = []
                
                if ragas_metrics:
                    # 执行Ragas评估
                    print(f"  执行Ragas评估...")
                    custom_llm = ChatTongyi(
                        model_name='qwen-plus',
                        temperature=0,
                        request_timeout=120,
                    )

                    embeddings = DashScopeEmbeddings()
                    ragas_result = evaluate(
                        dataset=eval_dataset,
                        metrics=ragas_metrics,
                        llm=custom_llm,
                        embeddings=embeddings
                    )
                    
                    # 提取评估结果并计算均值
                    if hasattr(ragas_result, 'to_pandas'):
                        df_ragas = ragas_result.to_pandas()
                        print(f"  📊 Ragas DataFrame shape: {df_ragas.shape}")
                        print(f"  📊 Columns: {df_ragas.columns.tolist()}")
                        
                        # 对每个指标取平均值
                        if 'context_precision' in self.metrics and 'context_precision' in df_ragas.columns:
                            precision_series = df_ragas['context_precision']
                            precision_val = precision_series.mean()
                            results['metrics']['context_precision'] = float(precision_val) if not np.isnan(precision_val) else 0.0
                            results['metrics']['context_precision_list'] = precision_series.tolist()  # 保存完整列表
                            precision_list = precision_series.tolist()  # 用于CSV
                            print(f"  ✓ Context Precision: {results['metrics']['context_precision']:.4f}")
                            print(f"     详细值: {[f'{x:.4f}' for x in precision_series.tolist()]}")
                            
                        # if 'context_recall' in self.metrics and 'context_recall' in df_ragas.columns:
                        #     recall_series = df_ragas['context_recall']
                        #     recall_val = recall_series.mean()
                        #     results['metrics']['context_recall'] = float(recall_val) if not np.isnan(recall_val) else 0.0
                        #     results['metrics']['context_recall_list'] = recall_series.tolist()  # 保存完整列表
                        #     recall_list = recall_series.tolist()  # 用于CSV
                        #     print(f"  ✓ Context Recall: {results['metrics']['context_recall']:.4f}")
                        #     print(f"     详细值: {[f'{x:.4f}' for x in recall_series.tolist()]}")
                        
                        print(f"  ✓ Ragas评估完成")
                
                # 保存评估数据集到CSV（包含Ragas指标）
                if self.output_dir:
                    try:
                        # 转换contexts列表为字符串（因为CSV不支持列表）
                        csv_data = []
                        for i in range(len(questions)):
                            csv_data.append({
                                'question': questions[i],
                                'ground_truth': ground_truths[i],
                                'contexts': '\n\n'.join(contexts_list[i]),  # 用两个换行符分隔
                                'context_precision': precision_list[i] if precision_list[i] is not None else '',
                                'chunk_size': self.chunk_size,
                                'overlap': self.overlap,
                                'num_contexts': len(contexts_list[i]),
                                # 'context_recall': recall_list[i] if recall_list[i] is not None else '',
                            })
                        
                        df_csv = pd.DataFrame(csv_data)
                        csv_path = os.path.join(self.output_dir, 'ragas_eval_dataset.csv')
                        
                        # 追加写入模式（如果文件存在）
                        if os.path.exists(csv_path):
                            df_csv.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                        else:
                            df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
                        
                        print(f"  💾 评估数据集已保存: ragas_eval_dataset.csv (chunk_size={self.chunk_size}, overlap={self.overlap}, {len(questions)} 条)")
                    except Exception as e:
                        print(f"  ⚠️  保存CSV失败: {e}")
            
            except Exception as e:
                print(f"  ⚠️  Ragas评估出错: {e}")
                # 如果Ragas评估失败,使用备用简单计算
                results['metrics']['context_precision'] = self._fallback_context_precision(
                    test_qa_pairs, rag_system
                )
                results['metrics']['context_recall'] = self._fallback_context_recall(
                    test_qa_pairs, rag_system
                )
        
        # 4. 添加分块统计信息(如果提供)
        if chunked_docs:
            results['metrics']['num_chunks'] = len(chunked_docs)
            results['metrics']['avg_chunk_length'] = float(np.mean([len(c['text']) for c in chunked_docs]))
        
        return results
    
    def _fallback_context_precision(self, test_qa_pairs: List[Dict], rag_system) -> float:
        """
        备用Context Precision计算(简化版)
        检查检索到的chunk中有多少包含答案
        """
        total_retrieved = 0
        relevant_retrieved = 0
        
        for qa in test_qa_pairs:
            if not qa.get('answers'):
                continue
            
            question = qa['question']
            answers = qa['answers']
            
            retrieved_docs, _ = rag_system.retrieve(question)
            
            for doc in retrieved_docs:
                total_retrieved += 1
                # 检查chunk中是否包含任意答案
                doc_text = doc['text'].lower()
                if any(ans.lower() in doc_text for ans in answers if ans):
                    relevant_retrieved += 1
        
        return relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
    
    def _fallback_context_recall(self, test_qa_pairs: List[Dict], rag_system) -> float:
        """
        备用Context Recall计算(简化版)
        检查有多少问题的答案被检索到
        """
        found_count = 0
        total_count = 0
        
        for qa in test_qa_pairs:
            if not qa.get('answers'):
                continue
            
            total_count += 1
            question = qa['question']
            answers = qa['answers']
            
            retrieved_docs, _ = rag_system.retrieve(question)
            
            # 检查是否检索到包含答案的chunk
            for doc in retrieved_docs:
                doc_text = doc['text'].lower()
                if any(ans.lower() in doc_text for ans in answers if ans):
                    found_count += 1
                    break
        
        return found_count / total_count if total_count > 0 else 0.0
        # return found_count / total_count if total_count > 0 else 0.0


if __name__ == "__main__":
    # 测试评估器
    config = {
        'evaluation': {
            'metrics': ['context_precision', 'context_recall', 'retrieval_time']
        }
    }
    
    evaluator = RAGEvaluator(config)
    print("RAG Evaluator (Ragas版本) 初始化成功")
