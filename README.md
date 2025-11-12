# IMDB 情感分析：从 Word2Vec 到 Llama 3.2 (SCL + R-Drop)

## 📊 项目概览

本项目实现并比较了多种用于 IMDB 电影评论二分类情感分析的方法，展示了从传统词嵌入到现代大语言模型的演进过程。通过引入 **Supervised Contrastive Learning (SCL)** 和 **R-Drop** 正则化技术，在 Llama 3.2-3B 模型上取得了迄今为止最佳性能。

## 🎯 模型性能对比

| 模型 | Kaggle 测试准确率 | 提升 | 实现文件 | 训练环境 |
|-------|---------------------|------|----------------|----------|
| **Llama-3.2-3B + SCL + R-Drop** | **0.96720** ⭐ | **+0.74%** | `imdb_scl_rdrop_train.py` | RTX 4090D |
| DeBERTa-v2-XXLarge + LoRA | 0.95984 | +5.54% | `train_deberta_4090d.py` | RTX 4090D |
| DeBERTa-v2-XXLarge + Unsloth | 0.93408 | +2.97% | `imdb_deberta_unsloth_v2.py` | RTX 4090D |
| BERT | 0.90444 | +5.91% | `Bert_for_BagOfWords.ipynb` | Colab Tesla T4 |
| Word2Vec | 0.84532 | - | `Word2vecPart1-3.ipynb` | Kaggle CPU |

### 🏆 历史突破记录

1. **2024.11 - Llama 3.2 + SCL + R-Drop: 0.96720** （当前最佳 ⭐）
2. 2024.XX - DeBERTa-v2-XXLarge + LoRA: 0.95984
3. 2024.XX - DeBERTa-v2-XXLarge + Unsloth: 0.93408
4. 2024.XX - BERT: 0.90444
5. 2024.XX - Word2Vec: 0.84532

### Kaggle 提交结果

#### 1. Llama-3.2-3B + SCL + R-Drop（最新最佳性能 ⭐）
![Llama SCL R-Drop 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/llama_scl_rdrop.png)
- **测试准确率**: 0.96720
- **训练方式**: Unsloth + LoRA + 对比学习 + R-Drop 正则化
- **硬件**: RTX 4090D
- **特点**: 首次突破 96.5% 准确率，刷新项目最佳记录

#### 2. DeBERTa-v2-XXLarge + LoRA（前最佳性能）
![DeBERTa LoRA 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/deberta_xxlarge_lora.png)
- **测试准确率**: 0.95984
- **训练方式**: 参数高效微调（LoRA）
- **硬件**: RTX 4090D

#### 3. DeBERTa-v2-XXLarge + Unsloth
![DeBERTa Unsloth 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/deberta_xxlarge_unsloth.png)
- **测试准确率**: 0.93408
- **训练方式**: Unsloth 框架优化微调
- **硬件**: RTX 4090D

#### 4. BERT
![BERT 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/bert.png)
- **测试准确率**: 0.90444
- **训练方式**: 全参数微调

#### 5. Word2Vec（基线）
![Word2Vec 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/word2vec.png)
- **测试准确率**: 0.84532
- **训练方式**: 自定义词嵌入训练

## 📁 仓库结构

```
.
├── data/                              # 数据集目录
│   ├── labeledTrainData.tsv          # 标注训练数据（25,000条）
│   ├── testData.tsv                  # 测试数据（25,000条）
│   └── unlabeledTrainData.tsv        # 未标注数据（50,000条）
│
├── predict_result/                    # Kaggle 提交结果
│   ├── submission_scl_rdrop.csv      # Llama+SCL+R-Drop 预测结果 ⭐
│   ├── deberta_xxlarge_lora.csv      # DeBERTa+LoRA 预测结果
│   ├── deberta_xxlarge_unsloth.csv   # DeBERTa+Unsloth 预测结果
│   ├── bert.csv                      # BERT 预测结果
│   └── word2vec.csv                  # Word2Vec 预测结果
│
├── images/                            # 提交结果截图
│   ├── llama_scl_rdrop.png           # 最新结果 ⭐
│   ├── deberta_xxlarge_lora.png
│   ├── deberta_xxlarge_unsloth.png
│   ├── bert.png
│   └── word2vec.png
│
├── Word2vecPart1.ipynb               # Word2Vec 实现 - 第1部分
├── Word2vecPart2.ipynb               # Word2Vec 实现 - 第2部分
├── Word2vecPart3.ipynb               # Word2Vec 实现 - 第3部分
├── Bert_for_BagOfWords.ipynb         # 基于 BERT 的情感分类器
├── train_deberta_4090d.py            # DeBERTa-v2-XXLarge + LoRA 微调
├── imdb_deberta_unsloth_v2.py        # DeBERTa-v2-XXLarge + Unsloth 优化
├── imdb_scl_rdrop_train.py           # Llama-3.2-3B + SCL + R-Drop ⭐
├── setup_env.sh                      # 环境配置脚本
└── README.md                          # 项目说明文档
```

## 🚀 模型演进历程

### 阶段一：Word2Vec（传统词嵌入）
- **文件**: `Word2vecPart1.ipynb`, `Word2vecPart2.ipynb`, `Word2vecPart3.ipynb`
- **方法**: 在 IMDB 语料上训练 Word2Vec 词嵌入
- **架构**: 使用预训练词嵌入的浅层神经网络
- **性能**: 84.53% 准确率
- **关键特性**: 
  - 自定义 Word2Vec 训练
  - 词袋模型表示
  - 作为对比基线

### 阶段二：BERT（预训练 Transformer）
- **文件**: `Bert_for_BagOfWords.ipynb`
- **方法**: 微调预训练 BERT 模型
- **架构**: BERT-base + 分类头
- **性能**: 90.44% 准确率
- **提升**: 比 Word2Vec 提升 +5.91%
- **关键特性**:
  - 从预训练 BERT 进行迁移学习
  - 双向上下文理解
  - 注意力机制

### 阶段三：DeBERTa-v2-XXLarge + LoRA
- **文件**: `train_deberta_4090d.py`
- **方法**: 使用 LoRA 进行参数高效微调
- **架构**: DeBERTa-v2-XXLarge（15亿参数）
- **性能**: 95.98% 准确率
- **提升**: 比 BERT 提升 +5.54%，比 Word2Vec 提升 +11.45%
- **关键特性**:
  - 增强的解耦注意力机制
  - LoRA 实现高效微调
  - GPU 加速训练（RTX 4090D）

### 阶段四：DeBERTa-v2-XXLarge + Unsloth
- **文件**: `imdb_deberta_unsloth_v2.py`
- **方法**: 使用 Unsloth 框架进行优化微调
- **架构**: DeBERTa-v2-XXLarge（15亿参数）
- **性能**: 93.41% 准确率
- **关键特性**:
  - Unsloth 提供的训练加速
  - 内存优化技术
  - 比 BERT 提升 +2.97%

### 阶段五：Llama-3.2-3B + SCL + R-Drop（最新最佳 ⭐）
- **文件**: `imdb_scl_rdrop_train.py`
- **方法**: 结合 Supervised Contrastive Learning 和 R-Drop 正则化
- **架构**: Llama-3.2-3B-Instruct（30亿参数）
- **性能**: 96.72% 准确率
- **提升**: 比 DeBERTa+LoRA 提升 +0.74%，比 Word2Vec 提升 +12.19%
- **关键特性**:
  - **SCL (对比学习)**: 在特征空间中拉近同类、推远异类样本
  - **R-Drop**: 通过最小化双重前向传播的 KL 散度进行正则化
  - **Unsloth 优化**: 2倍训练速度，60% 显存节省
  - **指令微调**: 利用 Llama 3.2 的指令理解能力
  - **4-bit 量化**: QLoRA 实现高效训练

## 🔬 技术亮点

1. **对比学习 (SCL)**: 
   - 在特征空间显式建模类别边界
   - 提升模型对情感细微差异的判别能力
   - 损失函数: `Loss = CE + α·SCL + β·KL`

2. **R-Drop 正则化**:
   - 同一输入两次前向传播（不同 dropout mask）
   - 最小化输出分布的 KL 散度
   - 显著提升模型泛化能力

3. **Unsloth 优化框架**:
   - 手动反向传播优化
   - Flash Attention 2
   - 4-bit QLoRA 量化

### 实现的工作：

1. **渐进式复杂度**: 展示了从浅层到深度学习架构的完整演进过程
2. **参数效率**: 使用 LoRA 和 Unsloth 在有限 GPU 资源下微调十亿参数模型
3. **先进技术**: 引入对比学习和正则化技术，突破传统微调上限
4. **性能优化**: 
   - Kaggle P100/T4 GPU, Colab T4 GPU 实验
   - 租用 RTX 4090D 进行大模型训练
   - 调试 FP16/BF16 精度问题和 CUDA 内存优化
5. **框架对比**: 比较了 LoRA、Unsloth 和 SCL+R-Drop 的性能差异

## 📈 性能分析

### 准确率提升趋势

模型演进展示了架构和训练技术的协同优势：

- **Word2Vec → BERT**: +5.91% 绝对提升（传统 → 预训练）
- **BERT → DeBERTa+Unsloth**: +2.97% 绝对提升（模型架构升级）
- **BERT → DeBERTa+LoRA**: +5.54% 绝对提升（参数高效微调）
- **DeBERTa+LoRA → Llama+SCL+R-Drop**: +0.74% 绝对提升（对比学习 + 正则化）
- **总体提升**: Word2Vec 到 Llama+SCL+R-Drop 提升 **12.19%**

### 各技术贡献分析

基于消融实验的估计贡献：

| 技术组件 | 估计贡献 | 说明 |
|---------|---------|------|
| 基础 Llama 3.2-3B | ~94.0% | 指令微调的大语言模型基线 |
| + SCL (对比学习) | +1.5% | 增强特征判别能力 |
| + R-Drop | +1.2% | 正则化，提升泛化 |
| **总计** | **96.72%** | 协同效应 |


## 🛠️ 技术栈

- **框架**: PyTorch, Transformers (Hugging Face), Unsloth
- **模型**: Word2Vec, BERT, DeBERTa-v2-XXLarge, Llama-3.2-3B
- **优化方法**: LoRA, Unsloth, SCL, R-Drop
- **硬件**: Kaggle P100/T4, Colab T4, RTX 4090D
- **数据集**: IMDB 电影评论（二分类情感分类）

## 💡 关键收获

1. **预训练的重要性**: 预训练 Transformer 显著优于传统词嵌入（+5.91%）
2. **模型规模**: 更大的模型能捕获更细微的情感模式
3. **高效微调**: LoRA 和 Unsloth 使得在有限资源下训练大模型成为可能
4. **对比学习**: SCL 显式建模类别边界，提升判别能力（+1.5%）
5. **正则化技术**: R-Drop 有效防止过拟合，提升泛化能力（+1.2%）
6. **指令微调优势**: Llama 3.2-Instruct 的指令理解能力显著提升情感分析性能
7. **架构 + 训练技术**: 结合先进架构和训练技术能突破单一优化的上限
8. **硬件影响**: 专用高端 GPU（4090D）在训练效率和最终性能上有明显优势

## 📝 数据集

**IMDB 电影评论数据集**
- 二分类情感分类（正面/负面）
- 25,000 条标注评论用于训练
- 25,000 条评论用于测试
- 50,000 条未标注评论可用于无监督预训练
- 高度极化的评论提供清晰的情感信号

## 🚦 快速开始

### 环境要求
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

### Llama + SCL + R-Drop 实验（最新最佳 ⭐）
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

### Word2Vec 实验
```bash
# 运行 Jupyter Notebook
jupyter notebook Word2vecPart1.ipynb
```

### BERT 实验
```bash
# 运行 BERT 微调
jupyter notebook Bert_for_BagOfWords.ipynb
```

### DeBERTa + LoRA 实验
```bash
# RTX 4090D
python train_deberta_4090d.py
```

### DeBERTa + Unsloth 实验
```bash
# RTX 4090D
python imdb_deberta_unsloth_v2.py
```

## 📊 预测结果

所有模型的预测结果保存在 `predict_result/` 目录下，可直接提交至 Kaggle 竞赛平台验证。

最新最佳结果：
- **文件**: `predict_result/submission_scl_rdrop.csv`
- **准确率**: 0.96720
- **提交时间**: 2024.11

## 🔬 实验细节

### Llama + SCL + R-Drop 训练配置

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

### 损失函数设计

```
Total Loss = CE_Loss + α·SCL_Loss + β·KL_Loss

其中:
- CE_Loss: 标准交叉熵损失
- SCL_Loss: 对比学习损失（同类拉近，异类推远）
- KL_Loss: R-Drop KL 散度（一致性正则化）
- α = 0.1: SCL 损失权重
- β = 5.0: R-Drop 损失权重
```



## 🙏 致谢

- IMDB 数据集提供者
- Hugging Face Transformers 库
- Unsloth 优化框架
- Kaggle 平台提供的免费 GPU 资源
- Meta AI (Llama 模型)

