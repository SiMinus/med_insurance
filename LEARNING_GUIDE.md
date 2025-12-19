# Chunking Strategy 学习指南

**学习时长**: 4小时 (分2天完成)  
**难度**: ⭐⭐⭐☆☆ (中级)  
**前置知识**: 完成向量模型对比实验

---

## 📅 学习计划概览

| 天数 | 时长 | 内容 | 产出 |
|------|------|------|------|
| **第1天** | 2小时 | 理论学习 + 环境准备 + 快速验证 | 理解chunking概念,运行3个实验 |
| **第2天** | 2小时 | 完整实验 + 结果分析 | 完成16个实验,生成分析报告 |

---

## 📚 第1天: 理论基础与快速验证 (2小时)

### ⏰ 时间分配
- **0:00-0:30** 理论学习
- **0:30-1:00** 环境准备
- **1:00-1:45** 快速验证实验(3个)
- **1:45-2:00** 第1天总结

---

### 📖 Session 1: 理论学习 (30分钟)

#### 1.1 什么是Chunking? (10分钟)

**定义**: 将长文档切分成小块(chunks)用于向量检索

**为什么需要Chunking?**
```
原始文档: 10000字的技术文档
问题: "如何配置Kubernetes网络策略?"

如果不分块:
❌ 整篇文档embedding → 语义混杂,检索不精确
❌ 答案埋在10000字中 → LLM处理成本高

分块后:
✅ 切成50个200字的chunks → 精确定位相关段落
✅ 只检索top-5相关chunks → 降低LLM成本
```

#### 1.2 Chunking的三个核心参数 (10分钟)

##### **参数1: Chunk Size (块大小)**

```
小块(200字):
  优点: 精确检索,噪声少
  缺点: 上下文不足,可能缺少关键信息

大块(800字):  
  优点: 上下文丰富,信息完整
  缺点: 噪声多,检索不精确

最优点: 取决于任务! (这就是实验要找的)
```

##### **参数2: Overlap (重叠度)**

```
无重叠(0%):
  Chunk 1: [0:200]
  Chunk 2: [200:400]  ← 可能截断关键信息

20%重叠(40字):
  Chunk 1: [0:200]
  Chunk 2: [160:360]  ← 保留上下文连贯性

50%重叠(100字):
  Chunk 1: [0:200]
  Chunk 2: [100:300]  ← 冗余大,但召回率高
```

**Trade-off**: 
- Overlap ↑ → 召回率 ↑ | 索引大小 ↑↑ | 检索时间 ↑

##### **参数3: Strategy (分块策略)**

**1. Fixed Size**: 固定字符数切分
```python
text = "这是第一句。这是第二句。这是第三句。"
chunks = ["这是第一句。这是", "第二句。这是第三"]
       ↑ 可能截断句子!
```

**2. Sentence-based**: 按句子边界切分
```python
text = "这是第一句。这是第二句。这是第三句。"
chunks = ["这是第一句。这是第二句。", "这是第三句。"]
       ↑ 保持语义完整!
```

**3. Semantic-based**: 基于语义相似度切分
```python
text = """
第一段讲Python基础。Python是编程语言。
第二段讲机器学习。深度学习很流行。
"""

# 语义分块会将相似内容聚合在一起
chunks = [
  "第一段讲Python基础。Python是编程语言。",  # 语义相关
  "第二段讲机器学习。深度学习很流行。"      # 语义相关
]
       ↑ 基于embedding相似度判断！
```

**三种策略对比**:
| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| Fixed | 速度快,简单 | 可能截断句子 | 对精确度要求不高 |
| Sentence | 语义完整 | 速度稍慢 | 一般问答系统 |
| Semantic | 语义连贯性最好 | 计算成本高 | 复杂推理任务 |

#### 1.3 本次实验要回答的问题 (10分钟)

**核心问题**:
1. ❓ 什么chunk_size在准确率和速度间最优?
2. ❓ overlap的边际收益是多少?20%够吗?50%值得吗?
3. ❓ Sentence策略比Fixed慢多少?收益多大?
4. ❓ Semantic策略的语义连贯性优势值得额外计算成本吗?
5. ❓ 如何根据数据特点选择chunking参数?

**预期发现**:
- Chunk Size: 400-600字符可能是最优区间
- Overlap: 20%性价比最高,50%边际收益递减
- Sentence: 准确率提升5-10%,速度慢20-30%
- Semantic: 准确率可能提升10-15%,但速度慢50%+

**✍️ 任务**: 在笔记本记录你的假设,实验后验证!

---

### 🛠️ Session 2: 环境准备 (30分钟)

#### 2.1 检查项目结构 (5分钟)

```bash
cd /Users/cuichenwei/Downloads/Project_Oct/Opt/chunking_strategy
ls -la
```

应该看到:
```
chunking_strategy/
├── config/
│   └── chunking_config.yaml
├── src/
│   ├── embeddings.py
│   ├── rag_system.py
│   ├── evaluator.py
│   └── data_loader_msmarco.py
├── main_chunking.py
├── data/
└── results/
```

#### 2.2 检查依赖 (10分钟)

```bash
# 检查Python版本(需要3.8+)
python --version

# 检查已安装的包
pip list | grep -E "transformers|sentence-transformers|faiss|datasets"
```

**如果缺少,安装**:
```bash
pip install transformers sentence-transformers faiss-cpu datasets pyyaml numpy matplotlib seaborn
```

#### 2.3 验证qwen3模型路径 (5分钟)

```bash
# 检查模型是否存在
ls /Users/cuichenwei/Downloads/Project_Nov/Qwen3-Embedding-0.6B
```

**如果不存在**, 修改 `config/chunking_config.yaml`:
```yaml
embedding_model:
  model_id: "sentence-transformers/all-MiniLM-L6-v2"  # 使用轻量级替代
  dimension: 384
```

#### 2.4 配置快速验证实验 (10分钟)

编辑 `config/chunking_config.yaml`:

```yaml
# 修改这些参数
dataset:
  subset_size: 10000  # 从50000降低(加快速度)
  num_queries: 1000   # 从5000降低
  
evaluation:
  test_size: 100  # 从1000降低

experiment:
  max_experiments: 3  # 只运行3个实验!
  skip_semantic: true # 第1天跳过语义分块(太慢)
```

**为什么减少数据?**
- 第1天目标: 验证代码能跑,理解流程
- 第2天再用完整数据集
- 语义分块计算成本高,第1天先跳过

---

### 🚀 Session 3: 快速验证实验 (45分钟)

#### 3.1 运行第一个实验 (30分钟)

```bash
cd /Users/cuichenwei/Downloads/Project_Oct/Opt/chunking_strategy
python main_chunking.py
```

**预计会看到**:
```
================================================================================
Chunking Strategy 对 RAG 检索效果的影响实验
================================================================================

加载MS MARCO数据集...
  - Passages: 10000
  - Queries: 1000 (测试集: 100)

加载MS MARCO数据集...
✓ 加载了 10000 个passages
✓ 加载了 1000 个queries
  - 训练集: 900
  - 测试集: 100

================================================================================
实验: fixed | Size=200 | Overlap=0
================================================================================

[1/4] 文档分块...
  完成分块: 10000 文档 → 25000 chunks (3.2s)
  平均chunk长度: 187.3 字符

[2/4] 构建向量索引...
正在加载模型: /Users/cuichenwei/Downloads/Project_Nov/Qwen3-Embedding-0.6B
模型加载完成, 维度: 1024
  索引构建完成: 25000 个文档, 耗时 45.8s

[3/4] 评估检索性能...

[4/4] 结果摘要:
  Chunks: 25000
  Accuracy@1: 0.4200
  Accuracy@5: 0.6800
  MRR: 0.5234
  检索时间: 0.0123s
  Context Precision: 0.5600
```

**⏰ 等待时间**: 每个实验约12-15分钟

#### 3.2 观察实验过程 (边等边学,15分钟)

**关注这些数字的变化**:

1. **Chunks数量**: 
   - Size=200 → ~25000 chunks
   - Size=400 → ~12500 chunks
   - Size=800 → ~6250 chunks

2. **Accuracy@1**:
   - 观察不同size的准确率
   - 哪个size最高?

3. **检索时间**:
   - chunks多 → 时间长?
   - 实际可能差异不大(FAISS很快)

**✍️ 任务**: 在表格中记录3个实验的结果

| Experiment | Size | Overlap | Chunks | Acc@1 | Acc@5 | Time |
|------------|------|---------|--------|-------|-------|------|
| 1          | 200  | 0       |        |       |       |      |
| 2          | 400  | 0       |        |       |       |      |
| 3          | 600  | 0       |        |       |       |      |

---

### 📝 Session 4: 第1天总结 (15分钟)

#### 4.1 回答这些问题:

1. **Chunking的作用是什么?**
   - 答: _______

2. **Chunk Size如何影响检索?**
   - 小块: _______
   - 大块: _______

3. **从今天的3个实验中,哪个size表现最好?**
   - 答: _______

4. **为什么需要overlap?**
   - 答: _______

#### 4.2 明天的准备:

- [ ] 恢复 `chunking_config.yaml` 中的数据量
  ```yaml
  dataset:
    subset_size: 50000
    num_queries: 5000
  evaluation:
    test_size: 1000
  experiment:
    max_experiments: 24  # 运行所有实验(含语义分块)
    skip_semantic: false # 运行语义分块实验
  ```

- [ ] 确保有足够时间(连续2-3小时,语义分块需要更多时间)

---

## 📊 第2天: 完整实验与分析 (2-3小时)

### ⏰ 时间分配
- **0:00-0:05** 回顾第1天
- **0:05-2:00** 运行完整实验(20个,含语义分块)
- **2:00-2:20** 结果分析
- **2:20-2:30** 总结与反思

---

### 🔄 Session 5: 回顾与准备 (5分钟)

#### 5.1 快速回顾

回答: 昨天发现chunk_size=___时准确率最高?

#### 5.2 确认配置

```bash
cd /Users/cuichenwei/Downloads/Project_Oct/Opt/2chunking_strategy

# 检查配置
cat config/chunking_config.yaml | grep -A2 "subset_size\|max_experiments\|skip_semantic"
```

应该是:
```yaml
subset_size: 50000
max_experiments: 24
skip_semantic: false  # 重要!要运行语义分块
```

---

### 🏃 Session 6: 运行完整实验 (115分钟)

#### 6.1 启动实验 (5分钟)

```bash
python main_chunking.py
```

**预计总时间**: ~110分钟(20个实验,语义分块较慢)

#### 6.2 实验进度监控 (边等边学)

**实验矩阵**:

**Fixed Size** (12个实验):
```
Size  | Overlap 0% | Overlap 20% | Overlap 50% |
------|------------|-------------|-------------|
200   | ✓ (5min)   | ✓ (6min)    | ✓ (8min)    |
400   | ✓ (4min)   | ✓ (5min)    | ✓ (7min)    |
600   | ✓ (4min)   | ✓ (5min)    | ✓ (7min)    |
800   | ✓ (4min)   | ✓ (5min)    | ✓ (7min)    |
```

**Sentence-based** (4个实验):
```
Target | Time  |
-------|-------|
200    | 6min  |
400    | 5min  |
600    | 5min  |
800    | 5min  |
```

**Semantic-based** (4个实验) - 🐌 最耗时:
```
Target | Overlap | Time   | 说明 |
-------|---------|--------|------|
400    | 0%      | 12min  | 需计算句子间相似度 |
400    | 20%     | 14min  | 语义分块+重叠处理 |
600    | 0%      | 10min  | chunk少,稍快 |
600    | 20%     | 12min  | |
```

⚠️ **注意**: 语义分块需要对每个句子计算embedding,所以特别慢!

#### 6.3 边等边思考 (填写预测)

**在实验运行时,预测结果**:

1. **哪个chunk_size准确率最高?**
   - 我的预测: _______
   - 原因: _______

2. **从0% → 20% → 50% overlap,准确率提升幅度?**
   - 0% → 20%: 提升 ____%
   - 20% → 50%: 提升 ____%

3. **Sentence vs Fixed,谁更好?**
   - 准确率差异: ____%
   - 时间差异: ____%

4. **Semantic vs Sentence,语义优势多大?**
   - 准确率提升: ____%
   - 时间成本: 增加____%
   - 值得吗: _______

---

### 📈 Session 7: 结果分析 (20分钟)

#### 7.1 查看结果文件 (5分钟)

```bash
cd results
ls -lt  # 查看最新生成的文件
```

你会看到:
```
chunking_results_20251107_143022.json   # 详细JSON
chunking_report_20251107_143022.txt     # 可读报告
heatmap_20251107_143022.png             # 热力图
```

#### 7.2 阅读文本报告 (10分钟)

```bash
cat chunking_report_*.txt
```

**关注这些指标**:

| 实验 | Chunks | Acc@1 | Acc@5 | MRR | Time |
|------|--------|-------|-------|-----|------|
| fixed, 200, 0 | | | | | |
| fixed, 200, 40 | | | | | |
| ... | | | | | |

**✍️ 任务**: 找出:
1. **最高Acc@1的配置**: _______
2. **最快检索的配置**: _______
3. **最佳性价比配置**: _______

#### 7.3 分析热力图 (5分钟)

打开 `heatmap_*.png`:

**观察**:
- X轴: Chunk Size (200, 400, 600, 800)
- Y轴: Overlap (0, 40/80/120/160, 100/200/300/400)
- 颜色: Accuracy@1 (越红越好)

**回答**:
1. 热点区域在哪?(最红的格子)
2. 是否存在明显的"最优区"?
3. Overlap增加时,颜色变化趋势?

---

### 🎓 Session 8: 总结与反思 (10分钟)

#### 8.1 核心发现

**发现1: Chunk Size影响**
```
实验结果:
  Size=200: Acc@1 = ____ (精确但上下文不足?)
  Size=400: Acc@1 = ____ (平衡点?)
  Size=600: Acc@1 = ____ (上下文丰富?)
  Size=800: Acc@1 = ____ (噪声增加?)

结论: 最优size = _____
```

**发现2: Overlap收益**
```
实验结果:
  0% overlap:  Acc@1 = ____
  20% overlap: Acc@1 = ____ (提升 ___%)
  50% overlap: Acc@1 = ____ (提升 ___%)

结论: ___% overlap性价比最高
```

**发现3: Sentence vs Fixed**
```
对比(以size=400为例):
  Fixed:    Acc@1 = ____ | Time = ____
  Sentence: Acc@1 = ____ | Time = ____
  
结论: Sentence策略_______
```

**发现4: Semantic vs Others**
```
对比(以size=400为例):
  Fixed:    Acc@1 = ____ | Time = ____
  Sentence: Acc@1 = ____ | Time = ____
  Semantic: Acc@1 = ____ | Time = ____ (慢___倍)
  
结论: Semantic策略适合_______场景
```

#### 8.2 实际应用建议

**场景1: 精确问答系统** (如客服FAQ)
```
推荐配置:
  - Chunk Size: _____ 
  - Overlap: _____%
  - Strategy: _______
原因: _______
```

**场景2: 文档总结** (需要更多上下文)
```
推荐配置:
  - Chunk Size: _____
  - Overlap: _____%
  - Strategy: _______
原因: _______
```

**场景3: 大规模检索** (速度优先)
```
推荐配置:
  - Chunk Size: _____
  - Overlap: _____%
  - Strategy: _______
原因: _______
```

**场景4: 复杂推理任务** (需要高度语义连贯)
```
推荐配置:
  - Chunk Size: _____
  - Overlap: _____%
  - Strategy: semantic (如果预算允许)
原因: _______
```

#### 8.3 回答初始问题

还记得第1天的4个核心问题吗?现在回答:

1. ❓ 什么chunk_size最优?
   - 答: _______

2. ❓ overlap的边际收益?
   - 答: _______

3. ❓ Sentence策略值得吗?
   - 答: _______

4. ❓ Semantic策略适合什么场景?
   - 答: _______

5. ❓ 如何选择chunking参数?
   - 答: _______

---

## 🎯 学习成果检验

### 完成这些,说明你已掌握:

- [ ] 理解chunking的作用和重要性
- [ ] 知道chunk_size如何影响精确率和召回率
- [ ] 理解overlap的成本/收益权衡
- [ ] 能根据场景选择合适的chunking策略
- [ ] 会用实验数据驱动RAG优化决策
- [ ] 理解RAG系统的预处理优化方法

---

## 📚 扩展学习 (可选)

如果时间充裕,尝试:

### 扩展1: 测试自己的数据
```bash
# 准备你的文档
documents = [
    {"id": 0, "text": "你的文档内容..."},
    ...
]

# 使用最优配置测试
from src.rag_system import TextChunker
chunks = TextChunker.chunk_documents(
    documents, 
    chunk_size=最优size,  # 从实验中获得
    overlap=最优overlap,
    strategy="sentence"  # 或 "fixed"
)
```

### 扩展2: 调整评估指标
```yaml
# config/chunking_config.yaml
evaluation:
  metrics:
    - "accuracy@k"
    - "context_precision"
    - "latency_per_token"  # 新增: 每token检索时间
```

### 扩展3: 尝试更多策略
- ✅ Semantic chunking (已在实验中!)
- Sliding window (滑动窗口)
- Hierarchical chunking (层级分块)
- Hybrid chunking (混合策略)

### 扩展4: 优化语义分块性能
```python
# 使用批处理加速语义分块
# 修改 src/rag_system.py 中的 chunk_text_semantic
# 一次性编码所有句子,而不是逐个编码
```

---

## 🐛 常见问题

### Q1: 实验时间太长怎么办?
**A**: 先用小数据集(subset_size=10000),理解原理后再用大数据集

### Q2: MS MARCO下载失败?
**A**: 代码会自动生成模拟数据,不影响学习chunking原理

### Q3: 内存不足?
**A**: 减少batch_size或使用更小的embedding模型

### Q4: 结果与预期差异大?
**A**: 正常!这就是实验的价值 - 用数据推翻/验证假设

---

## ✅ 学习清单

### 第1天完成:
- [ ] 理解chunking的3个核心参数
- [ ] 配置环境并运行3个验证实验
- [ ] 记录实验结果
- [ ] 准备第2天的完整实验

### 第2天完成:
- [ ] 运行16个完整实验
- [ ] 分析JSON和文本报告
- [ ] 解读热力图
- [ ] 总结核心发现
- [ ] 提出实际应用建议

---

**祝学习愉快! 🚀**

有问题随时查看 `README.md` 或代码注释!
