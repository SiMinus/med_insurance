"""
批量生成回答并使用 DeepEval 进行评估的脚本
基于 src/generate_answers.py
新增: faithfulness 和 answer correctness 评测
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import time

# --- 自动添加项目根目录到 sys.path ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
load_dotenv(project_root / ".env")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ------------------------------------

from src.basic_qa import KnowledgeBaseQA

import dashscope
from http import HTTPStatus

# --- DeepEval Imports ---
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM

class LocalQwen(DeepEvalBaseLLM):
    def __init__(self, qa_system):
        self.qa_system = qa_system

    def load_model(self):
        return self.qa_system.generator.model

    def generate(self, prompt: str) -> str:
        # 使用 chat template 包装 prompt
        # tokenizer = self.qa_system.generator.tokenizer
        # messages = [{"role": "user", "content": prompt}]
        # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 上面的方式可能导致 input 过长或者格式问题，deepeval 的 prompt 是纯文本指令
        # 我们可以直接传给 pipeline，但 Qwen Instruct 最好配合 ChatML
        
        # 尝试使用 apply_chat_template
        try:
            tokenizer = self.qa_system.generator.tokenizer
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = self.qa_system.generator(text, return_full_text=False)
            return output[0]["generated_text"]
        except Exception as e:
            print(f"Error in LocalQwen.generate: {e}")
            return "Error generating response."

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Qwen2.5-7B-Instruct"

class QwenPlusLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "qwen-plus"
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            # 尝试从 args 或者其他地方拿，这里简单抛错或打印
            print("Warning: DASHSCOPE_API_KEY not found in environment variables.")
        else:
            dashscope.api_key = api_key

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        try:
            response = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                result_format='message',
                temperature=0.01 # 降低随机性
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0]['message']['content']
            else:
                print(f"Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}")
                return "Error generating response."
        except Exception as e:
            print(f"Error calling Qwen Plus: {e}")
            return "Error generating response."

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def main():
    # 1. 路径配置
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "full_content.xlsx"
    output_dir = base_dir / "data" / "generated_answers_eval"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_result{timestamp}.csv"

    if not data_path.exists():
        print(f"错误: 数据文件不存在 {data_path}")
        return

    # 2. 读取 Excel 数据
    print(f"正在读取数据: {data_path}")
    try:
        df = pd.read_excel(data_path, engine='openpyxl')
        df.columns = [c.strip() for c in df.columns]

        col_map = {}
        for col in df.columns:
            if "问题" in col or "question" in col.lower():
                col_map["question"] = col
            if "标准答案" in col or "ground_truth" in col.lower():
                col_map["ground_truth"] = col
            if "来源" in col:
                col_map["source"] = col
        
        if "question" not in col_map or "ground_truth" not in col_map:
             print(f"错误: Excel 文件必须包含 '问题' 和 '标准答案' 列。当前列: {df.columns.tolist()}")
             return

        questions_data = df[col_map["question"]].tolist()
        ground_truths_data = df[col_map["ground_truth"]].tolist()
        sources_data = df[col_map["source"]].tolist() if "source" in col_map else [None] * len(questions_data)
        
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

    # 初始化 DeepEval 模型
    # 如果配置了 DASHSCOPE_API_KEY，优先使用 Qwen-Plus，否则使用本地模型
    if os.getenv("DASHSCOPE_API_KEY"):
        print("检测到 DASHSCOPE_API_KEY，使用 qwen-plus 进行评测...")
        deepeval_llm = QwenPlusLLM()
    else:
        print("未检测到 DASHSCOPE_API_KEY，使用本地 Qwen 模型进行评测 (速度较慢)...")
        deepeval_llm = LocalQwen(qa_system)

    # 4. 执行问答与检索
    print("开始执行问答与检索...")
    
    questions = []
    ground_truths = []
    answers = []
    contexts_list = []
    sources = []
    
    # Eval results
    faith_scores = []
    faith_reasons = []
    corr_scores = []
    corr_reasons = []
    
    valid_data = []
    for q, gt, src in zip(questions_data, ground_truths_data, sources_data):
        if pd.isna(q) or pd.isna(gt):
            continue
        valid_data.append((str(q).strip(), str(gt).strip(), src))
    
    # 设置生成的样本数量限制
    MAX_SAMPLES = 20
    if MAX_SAMPLES is not None:
        print(f"限制生成前 {MAX_SAMPLES} 条数据...")
        valid_data = valid_data[:MAX_SAMPLES]
    
    if not valid_data:
        print("没有有效样本，结束评估。")
        return

    questions_input = [item[0] for item in valid_data]
    ground_truths_input = [item[1] for item in valid_data]
    sources_input = [item[2] for item in valid_data]
    
    print(f"共 {len(questions_input)} 个问题，开始批量生成回答...")
    
    try:
        # 1. 批量生成回答
        results = qa_system.batch_answer(questions_input, batch_size=8)
        
        print("回答生成完毕，开始进行 DeepEval 评估...")
        
        # 2. 逐个评估
        pbar = tqdm(zip(questions_input, ground_truths_input, sources_input, results), total=len(questions_input), desc="Evaluating")
        for i, (q, gt, src, res) in enumerate(pbar):
            
            answer_text = res['answer']
            retrieved_contexts = [c.get('text', '') for c in res['contexts']]
            
            # 记录基本信息
            questions.append(q)
            ground_truths.append(gt)
            sources.append(src)
            answers.append(answer_text)
            
            # 格式化 Context 字符串
            ctx_items = []
            for c in res['contexts']:
                text = c.get('text', '')
                score = c.get('score', 'N/A')
                chunk_id = c.get('chunk_id', 'N/A')
                rank = c.get('rank', 0)
                original_rank = c.get('original_rank', '')
                ctx_items.append(f"{rank}({original_rank}).{text}\n(Score: {score}, ChunkID: {chunk_id})")
            ctx_str = '\n\n'.join(ctx_items)
            contexts_list.append(ctx_str)

            # DeepEval 评估
            try:
                test_case = LLMTestCase(
                    input=q,
                    actual_output=answer_text,
                    retrieval_context=retrieved_contexts,
                    expected_output=gt
                )
                
                # Faithfulness
                pbar.set_description(f"Eval {i+1}: Faithfulness")
                faith_metric = FaithfulnessMetric(
                    threshold=0.5,
                    model=deepeval_llm,
                    include_reason=True
                )
                faith_metric.measure(test_case, _show_indicator=False)
                faith_scores.append(faith_metric.score)
                faith_reasons.append(faith_metric.reason)
                alpha_steps = [
                    # 第一步：事实提取与召回率计算 (F_Score)
                    """1. [事实提取与召回率]: 
                        - 识别 'Expected Output' (标准答案) 中所有唯一的数值、百分比和关键实体，总数记为 N。
                        - 检查 'Actual Output' (生成回答) 中准确出现了其中多少项，总数记为 M。
                        - 计算基础事实分 F_Score = 0.3 + 0.7 * (M / N)。 (例如：标准答案有3个点，答对2个，则 F_Score = 0.3 + 0.7 * (2 / 3))。
                        - 特别规定：如果 M = 0 (即一个点都没对)，则 F_Score 直接设为 0。保留小数点后两位""",
                        
                    # 第二步：语义相似度计算 (S_Score)
                    """2. [语义相似度]: 
                        - 评估 'Actual Output' 与 'Expected Output' 在含义和意图上的语义相似程度。
                        - 忽略第一步中已经惩罚过的细节缺失，重点关注整体解释传达的信息是否一致。
                        - 给出一个 0.0 到 1.0 之间的分值作为 S_Score。保留小数点后两位""",

                    # 第三步：应用 Alpha 约束公式 (Alpha = 0.8)
                    """3. [Alpha 权重约束]: 使用以下公式计算最终得分：
                        Final_Score = (0.8 * F_Score) + (0.2 * S_Score)。保留小数点后两位""",
                    
                    # 第四步：兜底逻辑 (如果事实全错，强制低分)
                    """4. [最终审核]: 如果 F_Score 为 0，则无论回答多么礼貌或通顺，最终得分 Final_Score 绝对不能超过 0.3。记得用中文解释原因"""
                ]
                                # Answer Correctness (using GEval as replacement for AnswerCorrectnessMetric)
                pbar.set_description(f"Eval {i+1}: Correctness")
                corr_metric = GEval(
                    name="Correctness",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                    evaluation_steps=alpha_steps,
                    model=deepeval_llm,
                    threshold=0.5
                )
                corr_metric.measure(test_case, _show_indicator=False)
                corr_scores.append(corr_metric.score)
                corr_reasons.append(corr_metric.reason)
                
            except Exception as e:
                print(f"评估失败 (Index {i}): {e}")
                faith_scores.append(-1)
                faith_reasons.append(f"Error: {e}")
                corr_scores.append(-1)
                corr_reasons.append(f"Error: {e}")

    except Exception as e:
        print(f"批量处理或评估过程中断: {e}")
        import traceback
        traceback.print_exc()

    print(f"有效评估样本数: {len(questions)}")

    # 5. 保存结果
    if not questions:
        print("没有生成任何结果。")
        return

    result_df = pd.DataFrame({
        'question': questions,
        'ground_truth': ground_truths,
        'source_id': sources,
        'answer': answers,
        'contexts': contexts_list,
        'faithfulness_score': faith_scores,
        'faithfulness_reason': faith_reasons,
        'correctness_score': corr_scores,
        'correctness_reason': corr_reasons
    })
    
    try:
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {output_file}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")

if __name__ == "__main__":
    main()
