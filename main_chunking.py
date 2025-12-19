"""
Chunking Strategy实验主程序
研究不同文本分块策略对RAG检索效果的影响
"""

import os
import yaml
import json
import time
from datetime import datetime
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.embeddings import EmbeddingModelFactory
from src.rag_system import RAGSystem, TextChunker
from src.data_loader import load_squad_data
from src.evaluator import RAGEvaluator


def load_config(config_path: str = "config/chunking_config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_chunking_experiment(config: Dict, documents: List[Dict], test_queries: List[Dict],
                            chunk_size: int, overlap: int, strategy: str = "fixed", 
                            output_dir: str = None) -> Dict:
    """
    运行单个chunking实验
    
    Args:
        config: 配置字典
        passages: 文档列表
        test_queries: 测试查询列表
        chunk_size: chunk大小
        overlap: 重叠大小
        strategy: chunking策略
        output_dir: 输出目录（用于保存评估数据集）
        
    Returns:
        实验结果
    """
    print(f"\n{'='*80}")
    print(f"实验: {strategy} | Size={chunk_size} | Overlap={overlap}")
    print(f"{'='*80}")
    
    # 1. 创建embedding模型(固定使用qwen3-0.6b)
    model_config = config['embedding_model']
    # embedding_model = EmbeddingModelFactory.create_model(
    #     model_type=model_config['type'],
    #     model_id=model_config['model_id'],
    #     model_name=model_config['name']
    # )

    embedding_model = EmbeddingModelFactory.create_model(
        config=model_config
    )
    
    # 2. 文档分块
    print(f"\n[1/4] 文档分块...")
    start_time = time.time()
    
    if strategy == "fixed":
        chunked_docs = TextChunker.chunk_documents(
            documents, 
            chunk_size=chunk_size, 
            overlap=overlap,
            strategy="fixed"
        )
    elif strategy == "sentence":
        chunked_docs = TextChunker.chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=0,  # sentence策略不使用overlap
            strategy="sentence"
        )
    elif strategy == "semantic":
        chunked_docs = TextChunker.chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy="semantic",
            embedding_model=embedding_model  # 语义分块需要embedding模型
        )
    elif strategy == "recursive":
        chunked_docs = TextChunker.chunk_documents(
            documents,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy="recursive"
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    chunk_time = time.time() - start_time
    print(f"  完成分块: {len(documents)} 文档 → {len(chunked_docs)} chunks ({chunk_time:.2f}s)")
    print(f"  平均chunk长度: {np.mean([len(c['text']) for c in chunked_docs]):.1f} 字符")
    
    # 3. 构建RAG系统并索引
    print(f"\n[2/4] 构建向量索引...")
    rag_system = RAGSystem(embedding_model, config)
    index_time = rag_system.index_documents(chunked_docs)
    
    # 4. 评估
    print(f"\n[3/4] 评估检索性能...")
    evaluator = RAGEvaluator(config)
    
    # 设置输出目录和chunk配置（如果提供）
    if output_dir:
        evaluator.set_output_dir(output_dir)
    evaluator.set_chunk_config(chunk_size, overlap)
    
    eval_results = evaluator.evaluate_retrieval(rag_system, test_queries, chunked_docs)
    
    # 5. 汇总结果
    results = {
        'experiment_config': {
            'strategy': strategy,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'model': model_config['name']
        },
        'chunking_stats': {
            'num_original_docs': len(documents),
            'num_chunks': len(chunked_docs),
            'avg_chunk_length': float(np.mean([len(c['text']) for c in chunked_docs])),
            'chunk_time': chunk_time
        },
        'index_build_time': index_time,
        'metrics': eval_results['metrics']
    }
    
    # 打印关键指标
    print(f"\n[4/4] 结果摘要:")
    print(f"  Chunk Size: {chunk_size} | Overlap: {overlap}")
    print(f"  Chunks: {len(chunked_docs)}")
    print(f"  Context Precision: {results['metrics'].get('context_precision', 0):.4f}")
    # print(f"  Context Recall: {results['metrics'].get('context_recall', 0):.4f}")
    print(f"  检索时间: {results['metrics'].get('avg_retrieval_time', 0):.4f}s")
    
    return results


def run_all_experiments(config: Dict) -> List[Dict]:
    """运行所有实验"""
    
    # 加载数据
    print("\n" + "="*80)
    print("加载SQuAD v2数据集...")
    print("="*80)
    
    documents, train_queries, test_queries = load_squad_data(config)
    
    print(f"\n数据加载完成:")
    print(f"  Documents: {len(documents)}")
    print(f"  Test Queries: {len(test_queries)}")
    
    # 创建主输出目录（带时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_size = config['evaluation']['test_size']
    base_output_dir = config['output']['results_dir']
    main_output_dir = os.path.join(base_output_dir, f"{timestamp}_s{test_size}")
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"\n📁 主输出目录: {main_output_dir}")
    
    # 运行实验
    all_results = []
    
    for strategy_config in config['chunking_strategies']:
        strategy_name = strategy_config['name']
        
        # 跳过semantic策略(如果配置要求)
        if strategy_name == "semantic_based" and config['experiment'].get('skip_semantic', True):
            print(f"\n跳过 {strategy_name} 策略实验(配置要求)")
            continue
        
        # 只运行baseline实验(如果配置要求)
        if config['experiment'].get('run_baseline_only', False) and strategy_name != "fixed_size":
            print(f"\n跳过 {strategy_name} 策略实验(仅运行baseline)")
            continue
        
        for exp_config in strategy_config['experiments']:
            # 检查实验数量限制
            if len(all_results) >= config['experiment'].get('max_experiments', 20):
                print(f"\n达到最大实验数量限制({config['experiment']['max_experiments']})")
                break
            
            # 运行实验
            if strategy_name in ["fixed_size", "fixed"]:
                result = run_chunking_experiment(
                    config, documents, test_queries,
                    chunk_size=exp_config['chunk_size'],
                    overlap=exp_config['overlap'],
                    strategy="fixed",
                    output_dir=main_output_dir
                )
            elif strategy_name == "sentence_based":
                result = run_chunking_experiment(
                    config, documents, test_queries,
                    chunk_size=exp_config['target_size'],
                    overlap=0,
                    strategy="sentence",
                    output_dir=main_output_dir
                )
            elif strategy_name == "semantic_based":
                result = run_chunking_experiment(
                    config, documents, test_queries,
                    chunk_size=exp_config['target_size'],
                    overlap=exp_config.get('overlap', 0),
                    strategy="semantic",
                    output_dir=main_output_dir
                )
            elif strategy_name == "recursive_based":
                result = run_chunking_experiment(
                    config, documents, test_queries,
                    chunk_size=exp_config['target_size'],
                    overlap=exp_config.get('overlap', 0),
                    strategy="recursive",
                    output_dir=main_output_dir
                )
            else:
                continue
            
            all_results.append(result)
    
    return all_results, main_output_dir  # 返回结果和输出目录


def save_results(results: List[Dict], output_dir: str):
    """保存实验结果"""
    
    print(f"\n📁 结果保存到: {output_dir}")
    
    # 保存JSON结果
    json_file = os.path.join(output_dir, "results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 详细结果: results.json")
    
    # 生成文本报告
    report_file = os.path.join(output_dir, "report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Chunking Strategy 实验报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数: {len(results)}\n\n")
        
        f.write("实验结果:\n")
        f.write("-"*80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            exp = result['experiment_config']
            metrics = result['metrics']
            
            f.write(f"{i}. {exp['strategy']} | Size={exp['chunk_size']} | Overlap={exp['overlap']}\n")
            f.write(f"   Chunks: {result['chunking_stats']['num_chunks']}\n")
            f.write(f"   Context Precision: {metrics.get('context_precision', 0):.4f}\n")
            
            # 打印详细precision值
            if 'context_precision_list' in metrics:
                precision_list = metrics['context_precision_list']
                f.write(f"     详细值 ({len(precision_list)}条): {[f'{x:.4f}' for x in precision_list]}\n")
            
            # f.write(f"   Context Recall: {metrics.get('context_recall', 0):.4f}\n")
            # 
            # # 打印详细recall值
            # if 'context_recall_list' in metrics:
            #     recall_list = metrics['context_recall_list']
            #     f.write(f"     详细值 ({len(recall_list)}条): {[f'{x:.4f}' for x in recall_list]}\n")
            
            f.write(f"   检索时间: {metrics.get('avg_retrieval_time', 0):.4f}s\n")
            
            f.write("\n")
    
    print(f"✓ 文本报告: report.txt")


def generate_visualizations(results: List[Dict], output_dir: str):
    """生成可视化图表"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # # 提取fixed策略的结果(用于热力图)
        # fixed_results = [r for r in results if r['experiment_config']['strategy'] == 'fixed']
        
        # if not fixed_results:
        #     print("没有fixed策略结果,跳过热力图")
        # else:
        #     # 准备数据
        #     sizes = sorted(set(r['experiment_config']['chunk_size'] for r in fixed_results))
        #     overlaps = sorted(set(r['experiment_config']['overlap'] for r in fixed_results))
        #     
        #     precision_data = np.zeros((len(overlaps), len(sizes)))
        #     recall_data = np.zeros((len(overlaps), len(sizes)))
        #     
        #     for r in fixed_results:
        #         size_idx = sizes.index(r['experiment_config']['chunk_size'])
        #         overlap_idx = overlaps.index(r['experiment_config']['overlap'])
        #         precision_data[overlap_idx][size_idx] = r['metrics'].get('context_precision', 0)
        #         recall_data[overlap_idx][size_idx] = r['metrics'].get('context_recall', 0)
        #     
        #     # 1. Context Precision热力图
        #     plt.figure(figsize=(10, 6))
        #     sns.heatmap(precision_data, annot=True, fmt='.3f', 
        #                xticklabels=sizes, yticklabels=overlaps,
        #                cmap='YlOrRd', cbar_kws={'label': 'Context Precision'})
        #     plt.xlabel('Chunk Size')
        #     plt.ylabel('Overlap')
        #     plt.title('Context Precision: Chunk Size vs Overlap (Fixed Strategy)')
        #     
        #     precision_file = os.path.join(output_dir, "heatmap_precision.png")
        #     plt.savefig(precision_file, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"✓ Context Precision热力图: heatmap_precision.png")
        #     
        #     # 2. Context Recall热力图
        #     plt.figure(figsize=(10, 6))
        #     sns.heatmap(recall_data, annot=True, fmt='.3f', 
        #                xticklabels=sizes, yticklabels=overlaps,
        #                cmap='YlGnBu', cbar_kws={'label': 'Context Recall'})
        #     plt.xlabel('Chunk Size')
        #     plt.ylabel('Overlap')
        #     plt.title('Context Recall: Chunk Size vs Overlap (Fixed Strategy)')
        #     
        #     recall_file = os.path.join(output_dir, "heatmap_recall.png")
        #     plt.savefig(recall_file, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"✓ Context Recall热力图: heatmap_recall.png")
        
        # 3. Context Precision折线图 - 所有策略
        plt.figure(figsize=(16, 6))
        
        # 为每个实验创建标签（包含所有策略）
        x_labels = []
        for r in results:
            exp = r['experiment_config']
            strategy = exp['strategy']
            chunk_size = exp['chunk_size']
            overlap = exp['overlap']
            
            if strategy == 'fixed':
                x_labels.append(f"fixed\n({chunk_size},{overlap})")
            elif strategy == 'sentence':
                x_labels.append(f"sentence\n(size:{chunk_size})")
            elif strategy == 'semantic':
                x_labels.append(f"semantic\n({chunk_size},{overlap})")
            elif strategy == 'recursive':
                x_labels.append(f"recursive\n({chunk_size},{overlap})")
            else:
                x_labels.append(f"{strategy}\n({chunk_size},{overlap})")
        
        x_positions = list(range(len(results)))
        precision_vals = [r['metrics'].get('context_precision', 0) for r in results]
        
        # 绘制折线，不同策略用不同颜色
        colors = []
        for r in results:
            strategy = r['experiment_config']['strategy']
            if strategy == 'fixed':
                colors.append('orangered')
            elif strategy == 'sentence':
                colors.append('steelblue')
            elif strategy == 'semantic':
                colors.append('green')
            elif strategy == 'recursive':
                colors.append('purple')
            else:
                colors.append('gray')
        
        plt.plot(x_positions, precision_vals, marker='o', linewidth=2, markersize=8, color='darkgray', alpha=0.5)
        plt.scatter(x_positions, precision_vals, c=colors, s=100, zorder=3)
        
        # 在每个点旁边标注数值
        for i, (x, y) in enumerate(zip(x_positions, precision_vals)):
            plt.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
                       textcoords='offset points', fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('(Strategy, Size, Overlap)', fontsize=12)
        plt.ylabel('Context Precision', fontsize=12)
        plt.title('Context Precision Across All Experiments', fontsize=14)
        plt.xticks(x_positions, x_labels, rotation=45, ha='right', fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orangered', label='Fixed'),
            Patch(facecolor='steelblue', label='Sentence'),
            Patch(facecolor='green', label='Semantic'),
            Patch(facecolor='purple', label='Recursive')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        plt.tight_layout()
        
        line_precision_file = os.path.join(output_dir, "line_precision.png")
        plt.savefig(line_precision_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Context Precision折线图: line_precision.png")
        
        # # 4. Context Recall折线图 - 所有策略
        # plt.figure(figsize=(16, 6))
        # 
        # recall_vals = [r['metrics'].get('context_recall', 0) for r in results]
        # 
        # plt.plot(x_positions, recall_vals, marker='s', linewidth=2, markersize=8, color='darkgray', alpha=0.5)
        # plt.scatter(x_positions, recall_vals, c=colors, s=100, zorder=3)
        # 
        # # 在每个点旁边标注数值
        # for i, (x, y) in enumerate(zip(x_positions, recall_vals)):
        #     plt.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), 
        #                textcoords='offset points', fontsize=9, ha='center',
        #                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        # 
        # plt.xlabel('(Strategy, Size, Overlap)', fontsize=12)
        # plt.ylabel('Context Recall', fontsize=12)
        # plt.title('Context Recall Across All Experiments', fontsize=14)
        # plt.xticks(x_positions, x_labels, rotation=45, ha='right', fontsize=9)
        # plt.grid(True, alpha=0.3)
        # plt.legend(handles=legend_elements, loc='upper left')
        # plt.tight_layout()
        # 
        # line_recall_file = os.path.join(output_dir, "line_recall.png")
        # plt.savefig(line_recall_file, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"✓ Context Recall折线图: line_recall.png")
        
    except ImportError:
        print("matplotlib未安装,跳过可视化")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Chunking Strategy 对 RAG 检索效果的影响实验")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    # 运行实验
    results, output_dir = run_all_experiments(config)
    
    # 保存结果
    save_results(results, output_dir)
    
    # 生成可视化
    if config['output'].get('generate_plots', True):
        generate_visualizations(results, output_dir)
    
    print("\n" + "="*80)
    print(f"实验完成! 共运行 {len(results)} 个实验")
    print("="*80)


if __name__ == "__main__":
    main()
