# IMDB 情感分析：从 Word2Vec 到 DeBERTa

## 📊 项目概览

本项目实现并比较了多种用于 IMDB 电影评论二分类情感分析的方法，展示了从传统词嵌入到现代大语言模型的演进过程。

## 🎯 模型性能对比

| 模型 | Kaggle 测试准确率 | 实现文件 | 训练环境 |
|-------|---------------------|----------------|----------|
| **DeBERTa-v2-XXLarge + LoRA** | **0.95984** | `train_deberta_4090d.py` | RTX 4090D |
| **DeBERTa-v2-XXLarge + Unsloth** | **0.93408** | `imdb_deberta_unsloth_v2.py` | RTX 4090D |
| **BERT** | **0.90444** | `Bert_for_BagOfWords.ipynb` | colab Tesla T4 |
| **Word2Vec** | **0.84532** | `Word2vecPart1-3.ipynb` | Kaggle CPU |

### Kaggle 提交结果

#### 1. DeBERTa-v2-XXLarge + LoRA（最佳性能）
![DeBERTa LoRA 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/deberta_xxlarge_lora.png)
- **测试准确率**: 0.95984
- **训练方式**: 参数高效微调（LoRA）
- **硬件**: RTX 4090D

#### 2. DeBERTa-v2-XXLarge + Unsloth
![DeBERTa Unsloth 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/deberta_xxlarge_unsloth.png)
- **测试准确率**: 0.93408
- **训练方式**: Unsloth 框架优化微调
- **硬件**: RTX 4090D

#### 3. BERT
![BERT 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/bert.png)
- **测试准确率**: 0.90444
- **训练方式**: 全参数微调

#### 4. Word2Vec（基线）
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
│   ├── word2vec.csv                  # Word2Vec 预测结果
│   ├── bert.csv                      # BERT 预测结果
│   ├── deberta_xxlarge_lora.csv      # DeBERTa+LoRA 预测结果
│   └── deberta_xxlarge_unsloth.csv   # DeBERTa+Unsloth 预测结果
│
├── images/                            # 提交结果截图
│   ├── word2vec.png
│   ├── bert.png
│   ├── deberta_xxlarge_lora.png
│   └── deberta_xxlarge_unsloth.png
│
├── Word2vecPart1.ipynb               # Word2Vec 实现 - 第1部分
├── Word2vecPart2.ipynb               # Word2Vec 实现 - 第2部分
├── Word2vecPart3.ipynb               # Word2Vec 实现 - 第3部分
├── Bert_for_BagOfWords.ipynb         # 基于 BERT 的情感分类器
├── train_deberta_4090d.py            # DeBERTa-v2-XXLarge + LoRA 微调
├── imdb_deberta_unsloth_v2.py        # DeBERTa-v2-XXLarge + Unsloth 优化
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

### 阶段三：DeBERTa-v2-XXLarge + LoRA（最先进）
- **文件**: `train_deberta_4090d.py`
- **方法**: 使用 LoRA 进行参数高效微调
- **架构**: DeBERTa-v2-XXLarge（15亿参数）
- **性能**: 95.98% 准确率
- **提升**: 比 BERT 提升 +5.54%，比 Word2Vec 提升 +11.45%
- **关键特性**:
  - 增强的解耦注意力机制
  - LoRA 实现高效微调
  - GPU 加速训练（RTX 4090D）

### 阶段四：DeBERTa-v2-XXLarge + Unsloth（高效优化）
- **文件**: `imdb_deberta_unsloth_v2.py`
- **方法**: 使用 Unsloth 框架进行优化微调
- **架构**: DeBERTa-v2-XXLarge（15亿参数）
- **性能**: 93.41% 准确率
- **关键特性**:
  - Unsloth 提供的训练加速
  - 内存优化技术
  - 比 BERT 提升 +2.97%

## 🔬 技术亮点

### 实现的工作：

1. **渐进式复杂度**: 展示了从浅层到深度学习架构的完整演进过程
2. **参数效率**: 使用 LoRA 和 Unsloth 在有限 GPU 资源下微调十亿参数模型
3. **性能优化**: 
   - Kaggle P100/T4 GPU, colab T4 GPU实验
   - 租用 RTX 4090D 进行大模型训练
   - 调试 FP16 溢出和 CUDA 内存问题
4. **框架对比**: 比较了原生 LoRA 和 Unsloth 框架的性能差异

## 📈 性能分析

### 准确率提升趋势

模型演进展示了现代架构的明显优势：

- **Word2Vec → BERT**: +5.91% 绝对提升（从传统到预训练）
- **BERT → DeBERTa+Unsloth**: +2.97% 绝对提升（模型架构升级）
- **BERT → DeBERTa+LoRA**: +5.54% 绝对提升（最佳配置）
- **总体提升**: Word2Vec 到 DeBERTa+LoRA 提升 **11.45%**



## 🛠️ 技术栈

- **框架**: PyTorch, Transformers (Hugging Face), Unsloth
- **模型**: Word2Vec, BERT, DeBERTa-v2-XXLarge
- **优化方法**: LoRA, Unsloth
- **硬件**: Kaggle P100/T4, colab, RTX 4090D
- **数据集**: IMDB 电影评论（二分类情感分类）

## 💡 关键收获

1. **预训练的重要性**: 预训练 Transformer 显著优于传统词嵌入（+5.91%）
2. **模型规模**: 更大的模型（DeBERTa-XXLarge）能捕获更细微的情感模式
3. **高效微调**: LoRA 和 Unsloth 使得在有限资源下训练大模型成为可能
4. **架构设计**: DeBERTa 的增强注意力机制提供了持续的性能改进
5. **硬件影响**: 专用高端 GPU（4090D）vs 通用 GPU（Kaggle）在最终性能上有明显差异
6. **框架选择**: 根据硬件条件选择合适的微调框架（LoRA vs Unsloth）很重要

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
transformers >= 4.30
peft
unsloth
pandas
numpy
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

## 🙏 致谢

- IMDB 数据集提供者
- Hugging Face Transformers 库
- Unsloth 优化框架
- Kaggle 平台提供的免费 GPU 资源

## 📄 许可证

MIT License

---