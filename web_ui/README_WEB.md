# 医保知识库问答系统 - Web 部署指南

本项目包含一个基于 FastAPI 的后端服务和一个基于 React (Vite) 的前端界面。

## 目录结构

```
med_insurance/
├── src/
│   ├── server.py          # 后端 API 服务
│   └── basic_qa.py        # 问答核心逻辑
├── web_ui/                # 前端 React 项目
│   ├── src/
│   ├── package.json
│   └── vite.config.js
└── ...
```

## 1. 环境准备

### 后端依赖
确保已安装 Python 环境，并安装以下依赖：
```bash
pip install fastapi uvicorn
# 如果尚未安装其他依赖
pip install qdrant-client sentence-transformers transformers torch accelerate
```

### 前端依赖
需要安装 Node.js (建议 v16+)。
检查是否安装：
```bash
node -v
npm -v
```
如果未安装，请访问 [Node.js 官网](https://nodejs.org/) 下载安装。

## 2. 启动后端服务

在 `med_insurance` 根目录下运行：

```bash
# 默认运行在 8000 端口
python src/server.py
```

启动成功后，你会看到类似以下的日志：
```
INFO:     Started server process [pid]
INFO:     Waiting for application startup.
正在初始化知识库问答系统...
系统初始化完成！
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 3. 启动前端界面

打开一个新的终端窗口，进入 `web_ui` 目录：

```bash
cd web_ui

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

启动成功后，终端会显示访问地址，通常是：
```
  ➜  Local:   http://localhost:5173/
```

## 4. 使用

1. 打开浏览器访问 `http://localhost:5173`。
2. 在输入框中输入问题，例如："哈尔滨退休人员三级医院住院怎么报销？"。
3. 系统会流式输出回答，并列出参考的文档来源。

## 注意事项

- **显存占用**：后端服务启动时会加载 Embedding 模型和 LLM 模型，请确保有足够的显存（约 16GB+ 对于 Qwen2.5-7B）。
- **网络配置**：前端默认配置代理 `/api` 到 `http://localhost:8000`。如果后端运行在不同机器或端口，请修改 `web_ui/vite.config.js` 中的 `proxy` 配置。
- **模型路径**：后端默认自动查找本地模型。如果路径不正确，请修改 `src/server.py` 或 `src/basic_qa.py` 中的路径配置。
