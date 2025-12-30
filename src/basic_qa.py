"""基本的本地知识库问答脚本

- 向量检索: 使用本地 `qdrant_storage` 下的集合 `medical_insurance_chunks`
- 嵌入模型: 使用本地缓存的 BAAI/bge-m3
- 生成模型: 使用本地 `models/Qwen/Qwen2___5-7B-Instruct`

运行示例:
    python -m src.basic_qa --question "哈尔滨退休人员三级医院住院怎么报销?"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Union
from threading import Thread
from tqdm import tqdm # Add tqdm import

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_QDRANT_PATH = BASE_DIR / "qdrant_storage"
DEFAULT_COLLECTION = "medical_insurance_chunks"

# 先尝试仓库同级的 hf_cache，再尝试项目内的 hf_cache，最后回退到模型ID
_EMB_CANDIDATES = [
    REPO_ROOT
    / "hf_cache"
    / "hub"
    / "models--BAAI--bge-m3"
    / "snapshots"
    / "5617a9f61b028005a4858fdac845db406aefb181",
    BASE_DIR
    / "hf_cache"
    / "hub"
    / "models--BAAI--bge-m3"
    / "snapshots"
    / "5617a9f61b028005a4858fdac845db406aefb181",
]
DEFAULT_EMBEDDING_PATH: Union[str, Path] = next(
    (p for p in _EMB_CANDIDATES if p.exists()),
    "BAAI/bge-m3",
)

DEFAULT_LLM_PATH = REPO_ROOT / "models" / "Qwen" / "Qwen2___5-7B-Instruct"


def resolve_embedding_path(user_path: Union[str, Path, None]) -> Union[str, Path]:
    if user_path:
        p = Path(user_path)
        return p if p.exists() else str(user_path)
    return DEFAULT_EMBEDDING_PATH


def load_embedding_model(model_path: Union[str, Path]) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(str(model_path), device="cuda")


def load_llm(model_path: Union[str, Path]):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    import gc
    before = len(gc.get_objects())
    model = PeftModel.from_pretrained(model, "../fine-tuning/output_qwen_lora")
    after = len(gc.get_objects())
    after_before = after - before
    print(f"{after_before} 增量很小（仅 LoRA 参数 + 少量 wrapper 对象）")  # 增量很小（仅 LoRA 参数 + 少量 wrapper 对象）

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="cuda",
        max_new_tokens=512,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id, # 显式指定 EOS token ID
        # Removed invalid flags: temperature, top_p, top_k
    )


class KnowledgeBaseQA:
    def __init__(
        self,
        qdrant_path: Path = DEFAULT_QDRANT_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_path: Union[str, Path, None] = DEFAULT_EMBEDDING_PATH,
        llm_path: Union[str, Path] = DEFAULT_LLM_PATH,
        top_k: int = 4,
    ) -> None:
        self.client = QdrantClient(path=str(qdrant_path))
        self.collection_name = collection_name
        resolved_path = resolve_embedding_path(embedding_path)
        self.embedder = load_embedding_model(resolved_path)
        self.generator = load_llm(llm_path)
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict]:
        query_vector = self.embedder.encode(question, show_progress_bar=False)
        k = top_k or self.top_k
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=k,
        ).points

        contexts = []
        for idx, point in enumerate(results, start=1):
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            contexts.append(
                {
                    "rank": idx,
                    "score": point.score,
                    "text": payload.get("text", ""),
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                }
            )
        return contexts

    @staticmethod
    def build_prompt(question: str, contexts: List[Dict]) -> str:
        if contexts:
            ctx_block = "\n\n".join(
                f"[{c['rank']}] {c['text']}" for c in contexts
            )
        else:
            ctx_block = "无检索到的参考内容。"

        instruction = (
            "你是医保问答助手，请用中文结合参考内容简洁回答用户问题。但不要在答案中指明用的哪个参考内容, 不要有任何的与参考内容相关的编号出现"
            "切记只回答和问题相关的内容，不要重复回答相同内容，不要疯狂反复问好祝福，不要做复读机。和问题相关的内容回答完就停止"
        )

        return (
            f"{instruction}\n\n"
            f"参考内容:\n{ctx_block}\n\n"
            f"用户问题: {question}\n"
            "请给出回答。"
        )

    def answer(self, question: str, top_k: int | None = None) -> Dict:
        contexts = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, contexts)
        generation = self.generator(prompt, return_full_text=False)[0]["generated_text"]
        return {"answer": generation, "contexts": contexts}

    def batch_answer(self, questions: List[str], top_k: int | None = None, batch_size: int = 4) -> List[Dict]:
        """批量回答问题，提高 GPU 利用率"""
        # 1. 批量计算 Embeddings
        print("正在批量检索...")
        embeddings = self.embedder.encode(questions, show_progress_bar=True, batch_size=16)
        
        all_contexts = []
        k = top_k or self.top_k
        
        # 2. 检索上下文
        for vec in embeddings:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=vec.tolist(),
                limit=k,
            ).points
            
            contexts = []
            for idx, point in enumerate(results, start=1):
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                contexts.append(
                    {
                        "rank": idx,
                        "score": point.score,
                        "text": payload.get("text", ""),
                        "filename": metadata.get("filename"),
                        "page": metadata.get("page"),
                        "chunk_id": metadata.get("chunk_id"),
                    }
                )
            all_contexts.append(contexts)
            
        # 3. 构建 Prompts
        prompts = [self.build_prompt(q, ctx) for q, ctx in zip(questions, all_contexts)]
        
        # 4. 批量生成
        print(f"正在批量生成回答 (总数: {len(prompts)}, batch_size={batch_size})...")
        
        results = []
        # 手动分批处理以显示进度
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i : i + batch_size]
            try:
                # 注意: pipeline 内部也会处理 batching，但这里我们手动分块以便于显示进度和控制显存
                batch_outputs = self.generator(batch_prompts, batch_size=batch_size, return_full_text=False)
                
                for j, output in enumerate(batch_outputs):
                    ans = output[0]['generated_text']
                    original_idx = i + j
                    results.append({"answer": ans, "contexts": all_contexts[original_idx]})
            except Exception as e:
                print(f"Batch {i} generation failed: {e}")
                # 填充错误信息，保持列表长度一致
                for j in range(len(batch_prompts)):
                    results.append({"answer": "Error generating answer", "contexts": all_contexts[i+j]})
            
        return results

    def stream_answer(self, question: str, top_k: int | None = None):
        """流式回答生成"""
        contexts = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, contexts)
        
        # Yield contexts first as a special event or just keep them for the end?
        # For simplicity, we can yield a JSON string with contexts first, or just the text.
        # But usually streaming is just text. 
        # Let's yield the contexts as the first item (if the caller can handle it) 
        # or just yield text and let the caller retrieve contexts separately if needed.
        # Better: Yield a dict or specific structure. 
        # But for simple text streaming, let's just yield text chunks.
        # We can yield the contexts in the first chunk or a separate method.
        
        # Let's return the contexts and the generator
        streamer = TextIteratorStreamer(self.generator.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            text_inputs=prompt,
            return_full_text=False,
            streamer=streamer
        )
        
        thread = Thread(target=self.generator, kwargs=generation_kwargs)
        thread.start()
        
        return contexts, streamer


def main():
    parser = argparse.ArgumentParser(description="本地知识库问答")
    parser.add_argument("--question", required=True, help="用户问题")
    parser.add_argument("--top_k", type=int, default=4, help="检索返回的文档数")
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="自定义嵌入模型路径或模型ID(默认自动查找本地 bge-m3)",
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default=None,
        help="自定义本地生成模型路径，默认指向仓库同级 models/Qwen/Qwen2___5-7B-Instruct",
    )
    args = parser.parse_args()

    qa = KnowledgeBaseQA(
        top_k=args.top_k,
        embedding_path=args.embedding_path,
        llm_path=args.llm_path or DEFAULT_LLM_PATH,
    )
    result = qa.answer(args.question)

    print("\n=== 回答 ===")
    print(result["answer"])

    if result["contexts"]:
        print("\n=== 参考内容 ===")
        for ctx in result["contexts"]:
            source = f"{ctx['filename']} (p{ctx['page']})" if ctx.get("filename") else ""
            print(f"[{ctx['rank']}] score={ctx['score']:.4f} {source}\n{ctx['text']}\n")


if __name__ == "__main__":
    main()
