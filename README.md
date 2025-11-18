# 🎬 IMDB 电影评论情感分析 - 从 Word2Vec 到大语言模型

本项目实现并比较了多种用于 IMDB 电影评论二分类情感分析的方法，展示了从传统词嵌入到现代大语言模型的演进过程。通过引入 Supervised Contrastive Learning (SCL)、R-Drop 正则化技术以及 Instruction Tuning，在多个大语言模型上取得了优异性能。

## 📊 性能对比总览

| 模型 | Kaggle 测试准确率 | 提升 | 实现文件 | 训练环境 |
|------|------------------|------|----------|----------|
| **Llama-3.2-3B + SCL + R-Drop** | **0.96720** ⭐ | **+0.74%** | `imdb_scl_rdrop_train.py` | **RTX 4090D** |
| **Qwen3-4B + Instruction Tuning** | **0.96448** 🔥 | **+0.46%** | `imdb_qwen3_instruct.py` | **RTX 4090D** |
| DeBERTa-v2-XXLarge + LoRA | 0.95984 | +5.54% | `train_deberta_4090d.py` | RTX 4090D |
| DeBERTa-v2-XXLarge + Unsloth | 0.93408 | +2.97% | `imdb_deberta_unsloth_v2.py` | RTX 4090D |
| BERT | 0.90444 | +5.91% | `Bert_for_BagOfWords.ipynb` | Colab Tesla T4 |
| Word2Vec | 0.84532 | - | `Word2vecPart1-3.ipynb` | Kaggle CPU |

---

## 🏆 方法详解

### 🆕 **6. Qwen3-4B + Instruction Tuning (NEW)**

- **测试准确率**: 0.96448
- **训练方式**: Instruction Tuning + Unsloth + LoRA (4-bit)
- **硬件**: RTX 4090D
- **特点**: 
  - 首次在项目中应用指令微调范式
  - 使用 Alpaca prompt 模板进行情感分析
  - 性能仅次于 Llama+SCL+R-Drop，超越所有 DeBERTa 变体
  - 验证了指令微调在情感分析任务上的有效性

#### 关键技术亮点

**Instruction Tuning 范式**:
```
### Instruction:
Analyze the sentiment of the following movie review. 
Respond with 'positive' or 'negative'.

### Input:
{review_text}

### Response:
{sentiment_label}
```

**模型配置**:
- 基座模型: `Qwen3-4B-base` (40亿参数)
- 量化: 4-bit QLoRA
- LoRA 配置:
  - rank (r): 16
  - alpha: 32
  - dropout: 0
  - target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**训练参数**:
- Batch Size: 16
- Learning Rate: 2e-5
- Epochs: 3
- Optimizer: AdamW 8-bit
- Scheduler: Linear
- Max Sequence Length: 2048

**训练效率**:
- 训练时间: ~3-4 小时 (RTX 4090D)
- 显存占用: ~18GB (4-bit 量化)
- 推理速度: ~25,000 条评论/小时

---

### 1. Word2Vec 基线

- **测试准确率**: 0.84532
- **训练方式**: 自定义词嵌入训练
- 文件: `Word2vecPart1.ipynb`, `Word2vecPart2.ipynb`, `Word2vecPart3.ipynb`
- 方法: 在 IMDB 语料上训练 Word2Vec 词嵌入
- 架构: 使用预训练词嵌入的浅层神经网络
- 性能: 84.53% 准确率
- 关键特性:
  - 自定义 Word2Vec 训练
  - 词袋模型表示
  - 作为对比基线

### 2. BERT 微调

- **测试准确率**: 0.90444
- **训练方式**: 全参数微调
- 文件: `Bert_for_BagOfWords.ipynb`
- 方法: 微调预训练 BERT 模型
- 架构: BERT-base + 分类头
- 性能: 90.44% 准确率
- 提升: 比 Word2Vec 提升 +5.91%
- 关键特性:
  - 从预训练 BERT 进行迁移学习
  - 双向上下文理解
  - 注意力机制

### 3. DeBERTa-v2-XXLarge + LoRA

- **测试准确率**: 0.95984
- **训练方式**: 参数高效微调（LoRA）
- **硬件**: RTX 4090D
- 文件: `train_deberta_4090d.py`
- 方法: 使用 LoRA 进行参数高效微调
- 架构: DeBERTa-v2-XXLarge（15亿参数）
- 性能: 95.98% 准确率
- 提升: 比 BERT 提升 +5.54%，比 Word2Vec 提升 +11.45%
- 关键特性:
  - 增强的解耦注意力机制
  - LoRA 实现高效微调
  - GPU 加速训练（RTX 4090D）

### 4. DeBERTa-v2-XXLarge + Unsloth

- **测试准确率**: 0.93408
- **训练方式**: Unsloth 框架优化微调
- **硬件**: RTX 4090D
- 文件: `imdb_deberta_unsloth_v2.py`
- 方法: 使用 Unsloth 框架进行优化微调
- 架构: DeBERTa-v2-XXLarge（15亿参数）
- 性能: 93.41% 准确率
- 关键特性:
  - Unsloth 提供的训练加速
  - 内存优化技术
  - 比 BERT 提升 +2.97%

### 5. Llama-3.2-3B + SCL + R-Drop ⭐

- **测试准确率**: 0.96720
- **训练方式**: Unsloth + LoRA + 对比学习 + R-Drop 正则化
- **硬件**: RTX 4090D
- **特点**: 首次突破 96.5% 准确率，刷新项目最佳记录
- 文件: `imdb_scl_rdrop_train.py`
- 方法: 结合 Supervised Contrastive Learning 和 R-Drop 正则化
- 架构: Llama-3.2-3B-Instruct（30亿参数）
- 性能: 96.72% 准确率
- 提升: 比 DeBERTa+LoRA 提升 +0.74%，比 Word2Vec 提升 +12.19%
- 关键特性:
  - **SCL (对比学习)**: 在特征空间中拉近同类、推远异类样本
  - **R-Drop**: 通过最小化双重前向传播的 KL 散度进行正则化
  - **Unsloth 优化**: 2倍训练速度，60% 显存节省
  - **指令微调**: 利用 Llama 3.2 的指令理解能力
  - **4-bit 量化**: QLoRA 实现高效训练

#### SCL + R-Drop 详解

**对比学习 (SCL)**:
- 在特征空间显式建模类别边界
- 提升模型对情感细微差异的判别能力
- 损失函数: `Loss = CE + α·SCL + β·KL`

**R-Drop 正则化**:
- 同一输入两次前向传播（不同 dropout mask）
- 最小化输出分布的 KL 散度
- 显著提升模型泛化能力

**Unsloth 优化框架**:
- 手动反向传播优化
- Flash Attention 2
- 4-bit QLoRA 量化

---

## 📁 项目结构

```
.
├── data/                       # 数据集目录
│   ├── labeledTrainData.tsv   # 标注训练数据（25,000条）
│   ├── testData.tsv            # 测试数据（25,000条）
│   └── unlabeledTrainData.tsv # 未标注数据（50,000条）
│
├── predict_result/             # Kaggle 提交结果
│   ├── submission_scl_rdrop.csv        # Llama+SCL+R-Drop 预测结果 ⭐
│   ├── qwen3_4b_instruct_unsloth.csv   # Qwen3-4B Instruction 预测结果 🔥
│   ├── deberta_xxlarge_lora.csv        # DeBERTa+LoRA 预测结果
│   ├── deberta_xxlarge_unsloth.csv     # DeBERTa+Unsloth 预测结果
│   ├── bert.csv                        # BERT 预测结果
│   └── word2vec.csv                    # Word2Vec 预测结果
│
├── images/                     # 提交结果截图
│   ├── llama_scl_rdrop.png     # 最新结果 ⭐
│   ├── qwen_instruct.png       # Qwen3-4B 结果 🔥
│   ├── deberta_xxlarge_lora.png
│   ├── deberta_xxlarge_unsloth.png
│   ├── bert.png
│   └── word2vec.png
│
├── Word2vecPart1.ipynb         # Word2Vec 实现 - 第1部分
├── Word2vecPart2.ipynb         # Word2Vec 实现 - 第2部分
├── Word2vecPart3.ipynb         # Word2Vec 实现 - 第3部分
├── Bert_for_BagOfWords.ipynb   # 基于 BERT 的情感分类器
├── train_deberta_4090d.py      # DeBERTa-v2-XXLarge + LoRA 微调
├── imdb_deberta_unsloth_v2.py  # DeBERTa-v2-XXLarge + Unsloth 优化
├── imdb_qwen3_instruct.py      # Qwen3-4B + Instruction Tuning 🔥
├── imdb_scl_rdrop_train.py     # Llama-3.2-3B + SCL + R-Drop ⭐
├── setup_env.sh                # 环境配置脚本
└── README.md                   # 项目说明文档
```

---

## 🎯 项目特点

- **渐进式复杂度**: 展示了从浅层到深度学习架构的完整演进过程
- **参数效率**: 使用 LoRA 和 Unsloth 在有限 GPU 资源下微调十亿参数模型
- **先进技术**: 引入对比学习、正则化技术和指令微调，突破传统微调上限
- **性能优化**: 
  - Kaggle P100/T4 GPU, Colab T4 GPU 实验
  - 租用 RTX 4090D 进行大模型训练
  - 调试 FP16/BF16 精度问题和 CUDA 内存优化
- **框架对比**: 比较了 LoRA、Unsloth、SCL+R-Drop 和 Instruction Tuning 的性能差异

---

## 📈 性能演进分析

模型演进展示了架构和训练技术的协同优势：

- **Word2Vec → BERT**: +5.91% 绝对提升（传统 → 预训练）
- **BERT → DeBERTa+Unsloth**: +2.97% 绝对提升（模型架构升级）
- **BERT → DeBERTa+LoRA**: +5.54% 绝对提升（参数高效微调）
- **DeBERTa+LoRA → Qwen3-4B+Instruction**: +0.46% 绝对提升（指令微调范式）
- **Qwen3-4B → Llama+SCL+R-Drop**: +0.27% 绝对提升（对比学习 + 正则化）
- **总体提升**: Word2Vec 到 Llama+SCL+R-Drop 提升 12.19%

---

## 🛠️ 技术栈

- **框架**: PyTorch, Transformers (Hugging Face), Unsloth
- **模型**: Word2Vec, BERT, DeBERTa-v2-XXLarge, Qwen3-4B, Llama-3.2-3B
- **优化方法**: LoRA, Unsloth, SCL, R-Drop, Instruction Tuning
- **硬件**: Kaggle P100/T4, Colab T4, RTX 4090D
- **数据集**: IMDB 电影评论（二分类情感分类）

---

## 💡 核心发现

1. **预训练的重要性**: 预训练 Transformer 显著优于传统词嵌入（+5.91%）
2. **模型规模**: 更大的模型能捕获更细微的情感模式
3. **高效微调**: LoRA 和 Unsloth 使得在有限资源下训练大模型成为可能
4. **指令微调**: Instruction Tuning 范式在情感分析等 NLU 任务上表现优异（0.96448）
5. **对比学习**: SCL 显式建模类别边界，提升判别能力
6. **正则化技术**: R-Drop 有效防止过拟合，提升泛化能力
7. **指令理解优势**: 指令微调模型（Qwen3-4B, Llama 3.2-Instruct）的任务理解能力显著提升性能
8. **架构 + 训练技术**: 结合先进架构和训练技术能突破单一优化的上限
9. **硬件影响**: 专用高端 GPU 在训练效率和最终性能上有明显优势

---

## 📦 数据集信息

**IMDB 电影评论数据集**
- 二分类情感分类（正面/负面）
- 25,000 条标注评论用于训练
- 25,000 条评论用于测试
- 50,000 条未标注评论可用于无监督预训练
- 高度极化的评论提供清晰的情感信号

---

## 🚀 快速开始

### 环境配置

```bash
# 基础环境
python >= 3.8
torch >= 2.0
transformers >= 4.40
peft
unsloth
pandas
numpy
```

### 运行最新模型

#### Qwen3-4B Instruction Tuning 🔥

```bash
# RTX 4090D 训练
python imdb_qwen3_instruct.py

# 配置说明
# - Batch Size: 16
# - Learning Rate: 2e-5
# - Epochs: 3
# - LoRA r: 16, alpha: 32
# - Max Sequence Length: 2048
# - Training Time: ~3-4 小时 (4090D)
```

#### Llama 3.2 + SCL + R-Drop ⭐

```bash
# RTX 4090D 训练
python imdb_scl_rdrop_train.py

# 配置说明
# - Batch Size: 8
# - Gradient Accumulation: 2 (有效 batch=16)
# - Learning Rate: 2e-4
# - SCL Temperature: 0.07
# - SCL Weight: 0.1
# - R-Drop Alpha: 5.0
# - Training Time: ~4-5 小时 (4090D)
```

### 运行其他模型

```bash
# 运行 Jupyter Notebook
jupyter notebook Word2vecPart1.ipynb

# 运行 BERT 微调
jupyter notebook Bert_for_BagOfWords.ipynb

# RTX 4090D
python train_deberta_4090d.py

# RTX 4090D
python imdb_deberta_unsloth_v2.py
```

---

## 📊 Kaggle 提交

所有模型的预测结果保存在 `predict_result/` 目录下，可直接提交至 Kaggle 竞赛平台验证。

**最新最佳结果**：
- 文件: `predict_result/submission_scl_rdrop.csv`
- 准确率: 0.96720
- 提交时间: 2024.11

**最新指令微调结果** 🔥：
- 文件: `predict_result/qwen3_4b_instruct_unsloth.csv`
- 准确率: 0.96448
- 提交时间: 2024.11.18

---

## 🔧 Qwen3-4B 详细配置

### 模型配置

```python
# 模型配置
model_name = "unsloth/Qwen3-4B-base"
max_seq_length = 2048
load_in_4bit = True  # 4-bit 量化

# LoRA 配置
lora_r = 16
lora_alpha = 32
lora_dropout = 0
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 训练参数
batch_size = 16
num_epochs = 3
learning_rate = 2e-5
optimizer = "adamw_8bit"
lr_scheduler = "linear"
warmup_steps = 100
```

### Instruction Prompt Template

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of the following movie review. Respond with 'positive' or 'negative'.

### Input:
{review_text}

### Response:
{sentiment_label}"""
```

---

## 🔧 Llama 3.2 详细配置

### 模型配置

```python
# 模型配置
model_name = "unsloth/Llama-3.2-3B-Instruct"
max_seq_length = 512
load_in_4bit = True  # 4-bit 量化

# LoRA 配置
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05

# 训练参数
batch_size = 8
gradient_accumulation_steps = 2
num_epochs = 3
learning_rate = 2e-4

# SCL 参数
scl_temperature = 0.07
scl_weight = 0.1

# R-Drop 参数
rdrop_alpha = 5.0
```

### 损失函数

```
Total Loss = CE_Loss + α·SCL_Loss + β·KL_Loss
```

其中:
- **CE_Loss**: 标准交叉熵损失
- **SCL_Loss**: 对比学习损失（同类拉近，异类推远）
- **KL_Loss**: R-Drop KL 散度（一致性正则化）
- **α = 0.1**: SCL 损失权重
- **β = 5.0**: R-Drop 损失权重

---

## 🙏 致谢

- IMDB 数据集提供者
- Hugging Face Transformers 库
- Unsloth 优化框架
- Kaggle 平台提供的免费 GPU 资源
- Meta AI (Llama 模型)
- Alibaba Cloud (Qwen 模型)

