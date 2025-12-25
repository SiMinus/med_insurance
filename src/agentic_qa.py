from __future__ import annotations
import json
import re
import sys
from typing import List, Dict, Generator, Tuple, Any
from threading import Thread
from transformers import TextIteratorStreamer
from src.basic_qa import KnowledgeBaseQA

class AgenticQA:
    def __init__(self, basic_qa: KnowledgeBaseQA):
        self.qa = basic_qa
        self.tokenizer = self.qa.generator.tokenizer
        self.model = self.qa.generator.model
        
        # 定义工具
        self.tools_description = """
search_knowledge_base: 当需要回答关于医保政策的具体事实性问题时，调用此工具。输入参数为查询关键词。
"""
        self.tool_names = ["search_knowledge_base"]

    def search_knowledge_base(self, query: str) -> str:
        """工具实现: 检索知识库"""
        print(f"\n[Tool Call] search_knowledge_base('{query}')")
        # 复用 basic_qa 的检索功能
        contexts = self.qa.retrieve(query, top_k=3)
        if not contexts:
            return "未找到相关信息。"
        
        # 格式化返回内容
        return "\n".join([f"[{c['rank']}] {c['text']}" for c in contexts])

    def build_system_prompt(self) -> str:
        return f"""Answer the following questions as best you can. You have access to the following tools:

{self.tools_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{", ".join(self.tool_names)}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

    def stream_answer(self, question: str, top_k: int = 4):
        """
        Agentic RAG 的流式回答逻辑
        Yields:
            json string: {"type": "thought"|"chunk"|"contexts", "data": ...}
        """
        # 初始化 Prompt
        prompt = f"{self.build_system_prompt()}\n\nQuestion: {question}\n"
        history = prompt
        
        max_steps = 5
        step = 0
        final_answer_found = False
        
        # 打印初始日志
        print(f"\n=== Agentic RAG Start: {question} ===")

        while step < max_steps and not final_answer_found:
            step += 1
            
            # 准备生成
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # 设置停止词 (Observation:)
            # 注意: stop_strings 需要 transformers >= 4.39
            generation_kwargs = dict(
                text_inputs=history,
                return_full_text=False,
                streamer=streamer,
                max_new_tokens=512,
                stop_strings=["Observation:"], 
                tokenizer=self.tokenizer
            )
            
            # 在新线程中运行生成
            thread = Thread(target=self.qa.generator, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            buffer = ""
            
            # 消费流
            for new_text in streamer:
                generated_text += new_text
                
                # 如果已经进入 Answer 阶段，直接输出 chunk
                if final_answer_found:
                    yield json.dumps({"type": "chunk", "data": new_text}, ensure_ascii=False) + "\n"
                    continue

                buffer += new_text
                
                # 检查是否包含标记 (支持中英文冒号)
                match = re.search(r"Final Answer[:：]", buffer)
                if match:
                    # 找到了！
                    end_idx = match.end()
                    start_idx = match.start()
                    
                    # 分割: thought 部分 (标记之前)
                    thought_part = buffer[:start_idx]
                    # answer 部分 (标记之后)
                    answer_part = buffer[end_idx:]
                    
                    final_answer_found = True
                    
                    if thought_part:
                        yield json.dumps({"type": "thought", "data": thought_part}, ensure_ascii=False) + "\n"
                    
                    if answer_part:
                        yield json.dumps({"type": "chunk", "data": answer_part}, ensure_ascii=False) + "\n"
                    
                    buffer = "" # 清空 buffer
                else:
                    # 没找到完整标记，需要处理 buffer 滞留问题
                    # 只有当 buffer 结尾可能是标记的前缀时，才保留
                    # 标记: "Final Answer:" 或 "Final Answer："
                    # 简化处理：只检测 "Final Answer" 的前缀
                    
                    target = "Final Answer" # 不带冒号，先匹配这个主体
                    
                    match_len = 0
                    # 检查 buffer 结尾是否匹配 target 的前缀
                    # 限制检查长度，避免性能问题
                    check_len = min(len(buffer), len(target))
                    for i in range(check_len, 0, -1):
                        if target.startswith(buffer[-i:]):
                            match_len = i
                            break
                    
                    if match_len > 0:
                        # buffer 结尾匹配了前缀 (例如 "Final")
                        # 把前面的安全部分发出去
                        safe_part = buffer[:-match_len]
                        if safe_part:
                            yield json.dumps({"type": "thought", "data": safe_part}, ensure_ascii=False) + "\n"
                        # buffer 只保留匹配的部分
                        buffer = buffer[-match_len:]
                    else:
                        # 完全不匹配，全部发出去
                        # 但要注意：如果 buffer 是 "Final Answer" 但还没冒号，上面的逻辑会保留它
                        # 如果 buffer 是 "Final Answer Is" (不匹配)，这里会全部发出去
                        yield json.dumps({"type": "thought", "data": buffer}, ensure_ascii=False) + "\n"
                        buffer = ""

            # 本轮生成结束
            # 打印日志
            print(f"[Step {step}] Generated: {generated_text.strip()}")
            
            # 更新历史
            history += generated_text
            
            # 如果已经找到最终答案，结束循环
            if final_answer_found:
                break
            
            # 解析 Action
            # 期望格式: Action: search_knowledge_base\nAction Input: query
            action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", generated_text, re.DOTALL)
            
            if action_match:
                action_name = action_match.group(1).strip()
                action_input = action_match.group(2).strip()
                
                print(f"[Action] {action_name} -> {action_input}")
                
                observation = ""
                if action_name == "search_knowledge_base":
                    observation = self.search_knowledge_base(action_input)
                else:
                    observation = f"Error: Unknown tool '{action_name}'"
                
                print(f"[Observation] {observation[:100]}...") # 只打印前100字符
                
                # 构造 Observation 文本
                obs_text = f"\nObservation: {observation}\n"
                history += obs_text
                
                # 将 Observation 发送给前端显示在 Thought 区域
                yield json.dumps({"type": "thought", "data": obs_text}, ensure_ascii=False) + "\n"
                
            else:
                # 没找到 Action 也没找到 Final Answer，可能是生成中断或者格式错误
                # 强制添加一个 Observation 提示模型继续，或者结束
                if "Action:" in generated_text and "Action Input:" not in generated_text:
                     # 可能是生成了一半
                     pass
                elif "Observation:" in generated_text:
                     # 模型自己生成了 Observation?
                     pass
                else:
                     # 可能是模型不知道该干嘛了，或者已经结束了但没写 Final Answer
                     # 尝试强制结束
                     print("[Warning] Model loop without Action or Final Answer")
                     break

        print(f"=== Agentic RAG End ===\n")
