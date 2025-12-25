"""
Ragas 评估脚本
读取 data/page1_3.xlsx 中的问题和标准答案，使用 basic_qa.py 进行回答，
并使用 Ragas 评估 answer_correctness, faithfulness, context_recall 等指标。
结果保存至 data/ragas_result.csv
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv # Add this import

# --- 自动添加项目根目录到 sys.path ---
# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录 (src 的上一级)
project_root = current_file.parent.parent

# 加载 .env 文件
load_dotenv(project_root / ".env")

# 将根目录添加到 sys.path 的开头
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ------------------------------------

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_recall,
    context_precision,
)
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from src.basic_qa import KnowledgeBaseQA
import logging

# 禁用 tokenizers 警告和 ragas 日志
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger('ragas').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)

def main():
    # 1. 路径配置
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "page1_3.xlsx"
    output_dir = base_dir / "data"
    output_file = output_dir / "ragas_result.csv"

    if not data_path.exists():
        print(f"错误: 数据文件不存在 {data_path}")
        return

    # 2. 读取 Excel 数据
    print(f"正在读取数据: {data_path}")
    try:
        df = pd.read_excel(data_path)
        # 确保列名存在，去除空白字符
        df.columns = [c.strip() for c in df.columns]

        # --- 调试模式：只取前5条 ---
        # print("【调试模式】仅使用前 5 条数据进行评估...")
        # df = df.head(1)
        # -------------------------
        
        # 检查必要的列
        required_columns = ["问题", "标准答案"] # 根据用户描述调整列名匹配
        # 尝试匹配英文列名作为备选
        col_map = {}
        for col in df.columns:
            if "问题" in col or "question" in col.lower():
                col_map["question"] = col
            if "标准答案" in col or "ground_truth" in col.lower():
                col_map["ground_truth"] = col
        
        if "question" not in col_map or "ground_truth" not in col_map:
             print(f"错误: Excel 文件必须包含 '问题' 和 '标准答案' 列。当前列: {df.columns.tolist()}")
             return

        questions_data = df[col_map["question"]].tolist()
        ground_truths_data = df[col_map["ground_truth"]].tolist()
        
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # 3. 初始化 QA 系统
    print("正在初始化 QA 系统...")
    try:
        qa_system = KnowledgeBaseQA()
    except Exception as e:
        print(f"QA 系统初始化失败: {e}")
        return

    # 4. 执行问答与检索
    print("开始执行问答与检索...")
    
    questions = []
    ground_truths = []
    answers = []
    contexts_list = []
    
    # 预处理数据
    valid_data = []
    for q, gt in zip(questions_data, ground_truths_data):
        if pd.isna(q) or pd.isna(gt):
            continue
        valid_data.append((str(q).strip(), str(gt).strip()))
    
    if not valid_data:
        print("没有有效样本，结束评估。")
        return

    questions_input = [item[0] for item in valid_data]
    ground_truths_input = [item[1] for item in valid_data]
    
    print(f"共 {len(questions_input)} 个问题，开始批量处理...")
    
    try:
        # 使用 batch_answer 提高效率
        # batch_size 可根据显存大小调整，4090 可以尝试 8 或 16
        results = qa_system.batch_answer(questions_input, batch_size=16)
        
        for q, gt, res in zip(questions_input, ground_truths_input, results):
            questions.append(q)
            ground_truths.append(gt)
            answers.append(res['answer'])
            contexts_list.append([c['text'] for c in res['contexts']])
            
    except Exception as e:
        print(f"批量处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"有效样本数: {len(questions)}")

    # 5. 准备 Ragas 评估
    if not questions:
        print("没有有效样本，结束评估。")
        return

    # 构建 Dataset
    data_dict = {
        'question': questions,
        'answer': answers,
        'contexts': contexts_list,
        'ground_truth': ground_truths
    }
    eval_dataset = Dataset.from_dict(data_dict)

    # 配置评估指标
    # 用户要求: [answer_correctness, faithfulness, context_recall]
    # 注意: context_recall 需要 ground_truth
    ragas_metrics = [
        answer_correctness,
        faithfulness,
        context_recall,
        context_precision # 可选添加
    ]

    print("开始执行 Ragas 评估 (使用 qwen-plus)...")
    
    # 配置 LLM 和 Embeddings (使用 DashScope)
    # 请确保环境变量 DASHSCOPE_API_KEY 已设置
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("警告: 未检测到 DASHSCOPE_API_KEY 环境变量，评估可能会失败。")

    try:
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
        
        print("评估完成！")
        print(ragas_result)
        
        # 6. 保存结果
        # result_df = ragas_result.to_pandas()
        
        # 修复：ragas_result.to_pandas() 可能不包含 contexts，手动合并
        metrics_df = ragas_result.to_pandas()
        base_df = pd.DataFrame(data_dict)
        
        # 按索引合并，处理可能的重复列名
        # 如果 metrics_df 中已经包含了 question 等列，drop 掉避免重复
        cols_to_use = metrics_df.columns.difference(base_df.columns)
        result_df = pd.concat([base_df, metrics_df[cols_to_use]], axis=1)
        
        # 将 contexts 列表转换为字符串以便保存 CSV
        if 'contexts' in result_df.columns:
            result_df['contexts'] = result_df['contexts'].apply(lambda x: '\n\n'.join(x) if isinstance(x, list) else str(x))
        
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {output_file}")
        
    except Exception as e:
        print(f"Ragas 评估出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
