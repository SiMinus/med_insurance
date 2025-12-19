# Chunking Strategy 实验项目

研究文本分块策略(Chunking Strategy)对RAG检索效果的影响

## 🎯 项目目标

通过系统实验回答:
- ❓ 什么chunk size在准确率和速度间最优?
- ❓ overlap的边际收益是多少?
- ❓ Sentence策略比Fixed慢多少?收益多大?
- ❓ 如何根据数据特点选择chunking参数?

## 📁 项目结构

```
chunking_strategy/
├── config/
│   └── chunking_config.yaml     # 实验配置(16个实验)
├── src/
│   ├── embeddings.py            # Embedding模型封装
│   ├── rag_system.py            # RAG系统(含3种Chunker)
│   ├── evaluator.py             # 评估指标(含chunking专用)
│   └── data_loader_msmarco.py   # MS MARCO数据加载器
├── main_chunking.py             # 实验主程序
├── LEARNING_GUIDE.md            # 4小时学习指南(分2天)
├── data/                        # 数据缓存目录
└── results/                     # 实验结果输出
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install transformers sentence-transformers qdrant-client datasets pyyaml numpy matplotlib seaborn

# 进入项目目录
cd /Users/cuichenwei/Downloads/Project_Oct/Opt/chunking_strategy
```

### 2. 检查配置

编辑 `config/chunking_config.yaml`:

```yaml
# 检查模型路径
embedding_model:
  model_id: "/Users/cuichenwei/Downloads/Project_Nov/Qwen3-Embedding-0.6B"
  # 如果不存在,改为: "sentence-transformers/all-MiniLM-L6-v2"

# 第1天: 快速验证(10分钟)
dataset:
  subset_size: 10000  # 小数据集
experiment:
  max_experiments: 3  # 只运行3个

# 第2天: 完整实验(80分钟)
dataset:
  subset_size: 50000  # 完整数据集
experiment:
  max_experiments: 16  # 所有实验
```

### 3. 运行实验

```bash
python main_chunking.py
```

### 4. 查看结果

```bash
cd results
cat chunking_report_*.txt     # 文本报告
open heatmap_*.png            # 热力图
```

## 📊 实验矩阵

### Fixed Size Chunking (12个实验)

| Chunk Size | Overlap 0% | Overlap 20% | Overlap 50% |
|------------|-----------|-------------|-------------|
| 200        | ✓         | ✓           | ✓           |
| 400        | ✓         | ✓           | ✓           |
| 600        | ✓         | ✓           | ✓           |
| 800        | ✓         | ✓           | ✓           |

### Sentence-based Chunking (4个实验)

| Target Size | Max Sentences |
|-------------|---------------|
| 200         | 3             |
| 400         | 5             |
| 600         | 8             |
| 800         | 10            |

**共16个实验**, 预计耗时: ~80分钟(完整数据)

## 📈 评估指标

### 传统检索指标
- `accuracy@1/3/5`: 检索准确率
- `MRR`: 平均倒数排名
- `retrieval_time`: 检索耗时
- `index_build_time`: 索引构建时间

### Chunking专用指标
- `context_precision`: 检索块包含答案的比例
- `context_recall`: 答案覆盖率
- `avg_chunk_length`: 平均chunk长度
- `num_chunks`: 总chunk数量
- `chunk_utilization`: chunk利用率

## 🎓 学习指南

**推荐路径**: 按照 `LEARNING_GUIDE.md` 分2天学习

### 第1天 (2小时)
- 📖 理论学习: 理解chunking的3个核心参数
- 🛠️ 环境准备: 安装依赖,配置环境
- 🚀 快速验证: 运行3个实验,验证环境

### 第2天 (2小时)
- 🏃 完整实验: 运行16个实验
- 📊 结果分析: 分析报告,解读热力图
- 🎯 总结反思: 提炼发现,应用建议

## 💡 预期发现

基于经验,你可能会发现:

1. **Chunk Size**: 400-600字符是最优区间
2. **Overlap**: 20%性价比最高,50%边际收益递减
3. **Sentence**: 准确率提升5-10%,但速度慢20-30%

**但重点是**: 用实验数据验证或推翻这些假设!

## 🔧 配置说明

### 关键参数

```yaml
# 数据集大小(影响实验时长)
dataset:
  subset_size: 50000     # 5万文档
  num_queries: 5000      # 5千查询
  test_size: 1000        # 测试集1千

# 实验控制
experiment:
  run_baseline_only: false    # true=仅fixed实验
  skip_semantic: true         # 跳过semantic
  max_experiments: 16         # 限制实验数量

# Embedding模型(固定)
embedding_model:
  model_id: "qwen3-0.6b或minilm"
```

### 时间优化

如果时间有限:

```yaml
# 方案1: 减少数据量
dataset:
  subset_size: 10000  # 5万→1万
  
# 方案2: 减少实验数
experiment:
  max_experiments: 6  # 只测试关键配置
```

## 📊 结果示例

### 文本报告
```
1. fixed | Size=400 | Overlap=80
   Chunks: 125000
   Accuracy@1: 0.6234
   Accuracy@5: 0.8456
   MRR: 0.7123
   检索时间: 0.0156s
   Context Precision: 0.7834
```

### 热力图
- X轴: Chunk Size (200→800)
- Y轴: Overlap (0%→50%)
- 颜色: Accuracy@1 (越红越好)

## 🐛 常见问题

### Q1: MS MARCO下载失败?
**A**: 代码会自动生成模拟数据,不影响学习

### Q2: qwen3模型路径错误?
**A**: 改用 `sentence-transformers/all-MiniLM-L6-v2`

### Q3: 实验时间太长?
**A**: 设置 `subset_size: 10000` 和 `max_experiments: 6`

### Q4: 内存不足?
**A**: 减少 `subset_size` 到 20000 以下

## 📚 技术栈

- **Embeddings**: Qwen3-0.6B / MiniLM
- **Vector DB**: Qdrant
- **Dataset**: MS MARCO Passages
- **Visualization**: matplotlib, seaborn
- **Config**: YAML

## 🎯 学习成果

完成后你将能:
- ✅ 理解chunking对RAG系统的影响
- ✅ 根据场景选择最优chunking参数
- ✅ 用数据驱动RAG系统优化决策
- ✅ 权衡准确率、速度、成本

## 📄 License

MIT License

---

**开始学习吧! 🚀**

详细步骤请查看 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)
