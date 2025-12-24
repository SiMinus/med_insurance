import os
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Ensure we can import from src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.basic_qa import KnowledgeBaseQA

app = FastAPI(title="Medical Insurance QA API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global QA instance
qa_system = None

class ChatRequest(BaseModel):
    question: str
    top_k: int = 4

@app.on_event("startup")
async def startup_event():
    global qa_system
    print("正在初始化知识库问答系统...")
    # 使用默认路径，如果需要自定义可以通过环境变量或修改此处
    # 注意：这里会加载模型，需要显存
    try:
        qa_system = KnowledgeBaseQA()
        print("系统初始化完成！")
    except Exception as e:
        print(f"系统初始化失败: {e}")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not qa_system:
        raise HTTPException(status_code=503, detail="System is initializing or failed to initialize")

    try:
        contexts, streamer = qa_system.stream_answer(request.question, request.top_k)
        
        async def response_generator() -> AsyncGenerator[str, None]:
            # 1. 发送参考文档
            yield json.dumps({
                "type": "contexts",
                "data": contexts
            }, ensure_ascii=False) + "\n"
            
            # 2. 发送生成的文本流
            full_answer = ""
            for text in streamer:
                full_answer += text
                yield json.dumps({
                    "type": "chunk",
                    "data": text
                }, ensure_ascii=False) + "\n"
                await asyncio.sleep(0.01)  # 让出控制权
            
            # 3. 发送结束标记（可选，这里流结束就是结束）
            
        return StreamingResponse(response_generator(), media_type="application/x-ndjson")
        
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
