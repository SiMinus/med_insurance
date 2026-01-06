from __future__ import annotations
import json
from typing import List, Dict, Any
import torch
from src.basic_qa import KnowledgeBaseQA
import os
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn
)
# Set environment variable before importing transformers to suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

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
        contexts = self.qa.retrieve(query, top_k=4)
        if not contexts:
            return "未找到相关信息。"
        
        # 格式化返回内容
        return "\n".join([f"[{c['rank']}] {c['text']}" for c in contexts])

    def build_system_prompt(self) -> str:
        """系统提示词：鼓励先澄清再回答，使用工具检索。"""
        return (
            "你是医保政策助手，需要依据工具返回的事实来回答。"
            "当工具返回中有不止一个内容可以回答，针对的是不同场景有不同前提，必须向用户追问具体指的是那个场景"
            "（如病种、医院等级、本地/异地、在职/退休等），不要凭空猜测。"
            "可以使用以下工具获取资料：\n" + self.tools_description +
            "\n对话规则：\n"
            "- 工具调用格式由系统自动处理，你只需决定是否调用。\n"
            "- 如果仍然信息不足，继续追问，直到可给出明确答案。\n"
            "- 回答时用中文，简洁且基于检索到的内容。"
        )

    def _tool_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "search_knowledge_base",
                "description": "检索医保知识库，输入用户最近的两个问题，返回相关片段。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "必须包含用户最近两个问询中的所有关键限定条件，如：就医类型（住院/门诊）、医院等级（三级/二级）、参保类型（职工/居民）、就地类型（本地/异地）等。"}
                    },
                    "required": ["query"],
                },
            }
        ]

    def _extract_json_object(self, text: str) -> str | None:
        """从文本中提取第一个 JSON 对象（基于括号平衡），用于回退解析。"""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    def _build_search_query(self, messages: List[Dict[str, Any]]) -> str:
        """
        规则：
        - 如果上一个 assistant 是追问，则合并最近两个 user
        - 否则只用最近一个 user
        """
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
        if not user_msgs:
            return ""

        # 判断上一个 assistant 是否追问
        
        if assistant_msgs[-1].startswith("追问"):
            return "；".join(user_msgs[-2:])

        return user_msgs[-1]

    def _chat_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        """优先使用模型的 chat 接口；若不存在则回退到 generate 并解析 JSON 输出。

        返回值与原生 chat 兼容：要么是 dict（可能包含 function_call 和 content），
        要么是字符串 content。
        """

        # 回退：拼接一个明确的指令，要求模型以 JSON 输出 function_call 或 content
        tool_block = json.dumps(tools, ensure_ascii=False)
        # 将消息合并为文本提示（简化），保留 system 与 user 最近的上下文
        assembled = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            assembled.append(f"[{role}] {content}")
        prompt = "\n".join(assembled)
        prompt += (
            "\n\n可用工具：" + tool_block +
            "\n\n你必须只输出一个 JSON 对象，不要输出多余文字。JSON 必须包含字段：" 
            "\n- thought: 字符串。每次都要输出，说明【当前情况】和【下一步怎么做】。" 
            "\n- action: 对象。每次都要输出，用来表示你接下来要做什么。"
            "\n\naction 只有一下三种情况："
            "\n1) 需要调用工具：action = {\"type\": \"function_call\", \"name\": \"search_knowledge_base\", \"arguments\": {\"query\": \"...\"}}"
            "\n2) 需要追问用户：action = {\"type\": \"ask\", \"content\": \"你的追问\"}"
            "\n3) 可以直接回答：action = {\"type\": \"answer\", \"content\": \"你的最终答案\"}"
        )

        schema = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["function_call", "ask", "answer"]
                },
                "name": {"type": "string"},
                "arguments": {"type": "object"},
                "content": {"type": "string"}
            },
            "required": ["type"]
        }
    },
    "required": ["thought", "action"]
}


        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        parser = JsonSchemaParser(schema)

        prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer,
            parser
        )

        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只解码新生成部分（避免把 prompt/system 全部当作 content 打出来）
        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][input_len:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 尝试解析为 JSON（允许前后夹杂少量多余文本）
        json_blob = self._extract_json_object(decoded) or decoded
        try:
            return json.loads(json_blob)
        except Exception:
            # 回退为纯文本 content
            return decoded

    def stream_answer(self, question: str, top_k: int = 4, messages: List[Dict[str, Any]] | None = None):
        """基于 Qwen 函数调用的 Agent 循环，持续澄清后给出答案。"""
        if messages is None or len(messages) == 0:
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
            ]

        # Ensure system prompt exists at the beginning
        if messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.build_system_prompt()})

        # Always append current user question
        # messages.append({"role": "user", "content": question})
        print("===================================================")
        print(messages)
        print("===================================================")
        tools = self._tool_schema()
        max_turns = 8

        print(f"\n=== Agentic RAG Start: {question} ===")

        for turn in range(max_turns):
            # 1) 调用模型，通过包装器优先使用 chat，若无则回退到 generate
            response = self._chat_with_tools(messages, tools)
            print(f"\n response: {str(response)}")

            # 兼容两类返回：
            # - 原生 chat: dict 可能包含 function_call/content
            # - 回退 generate: dict 可能包含 thought/action
            if not isinstance(response, dict):

                    
                print(f"\n response不合格: {str(response)}")



            # 1) thought：每轮都尽量输出（若没有就不输出）
            thought = response.get("thought")
            if isinstance(thought, str) and thought.strip():
                yield json.dumps({"type": "thought", "data": thought}, ensure_ascii=False) + "\n"
            print(f"\n[Thought] {thought}")

            # 2) action / function_call / content
            function_call = None
            action = response.get("action")
            content = None

            # 回退 action 格式：function_call
            if isinstance(action, dict) and action.get("type") == "function_call":
                function_call = {
                    "name": action.get("name"),
                    "arguments": action.get("arguments", {}),
                }
                content = None

            # 回退 action 格式：ask/answer
            if isinstance(action, dict) and action.get("type") in {"ask", "answer"}:
                content = action.get("content")
                function_call = None

            if function_call:
                func_name = function_call.get("name")
                func_args = function_call.get("arguments", {})

                # arguments 可能是 JSON 字符串或 dict
                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except Exception:
                        func_args = {"query": func_args}

                print(f"\n[ToolCall] {func_name} args={func_args}")

                if func_name == "search_knowledge_base":
                    query = self._build_search_query(messages)
                    observation = self.search_knowledge_base(query)
                else:
                    # 对未知工具提供纠正提示
                    observation = (
                        f"系统不支持工具名 '{func_name}'，只支持以下工具："
                        f"{', '.join(self.tool_names)}。请更正后重试。"
                    )

                print(f"\n[Observation] {observation}")

                # 将工具返回作为 observation 放回 messages
                messages.append({
                    "role": "tool",  # 使用固定角色，避免未知工具名导致问题
                    "content": observation,
                })

                continue

            # 无工具调用，则认为给出了最终回答/追问
            if content:
                
                if action.get("type") == "answer":
                    messages.append({
                    "role": "assistant",
                    "content": content,
                })
                    yield json.dumps({"type": "chunk", "data": content}, ensure_ascii=False) + "\n"
                    print(f"\n[Answer] {content}")
                    
                else:
                    messages.append({
                    "role": "assistant",
                    "content": "追问:" + content,
                })
                    yield json.dumps({"type": "chunk", "data": content}, ensure_ascii=False) + "\n"
                    print(f"\n[Ask] {content}")

                break


        print(f"=== Agentic RAG End ===\n")
