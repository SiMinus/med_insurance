"""
批量生成回答脚本
读取 data/page1_3.xlsx 中的问题和标准答案，使用 basic_qa.py 进行回答。
结果保存至 data/generated_answers/generated_answers_YYYYMMDD_HHMMSS.csv
不进行 Ragas 评估。
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

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

from src.basic_qa import KnowledgeBaseQA

def main():
    # 1. 路径配置
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "page1_3.xlsx"
    output_dir = base_dir / "data" / "generated_answers"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"generated_answers_{timestamp}.csv"

    if not data_path.exists():
        print(f"错误: 数据文件不存在 {data_path}")
        return

    # 2. 读取 Excel 数据
    print(f"正在读取数据: {data_path}")
    try:
        df = pd.read_excel(data_path)
        # 确保列名存在，去除空白字符
        df.columns = [c.strip() for c in df.columns]

        # 检查必要的列
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
            # 将 contexts 列表转换为字符串以便保存 CSV
            ctx_str = '\n\n'.join([c['text'] for c in res['contexts']])
            contexts_list.append(ctx_str)
            
    except Exception as e:
        print(f"批量处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"有效样本数: {len(questions)}")

    # 5. 保存结果
    if not questions:
        print("没有生成任何结果。")
        return

    result_df = pd.DataFrame({
        'question': questions,
        'ground_truth': ground_truths,
        'answer': answers,
        'contexts': contexts_list
    })
    
    try:
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {output_file}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")

if __name__ == "__main__":
    main()
