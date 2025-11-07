# IMDB 情感分析：从 Word2Vec 到 DeBERTa

## 📊 项目概览

本项目实现并比较了多种用于 IMDB 电影评论二分类情感分析的方法.

## 🎯 模型性能对比

| 模型 | Kaggle 测试准确率 | 实现文件 |
|-------|---------------------|----------------|
| **DeBERTa-v2-XXLarge + LoRA** | **0.95984** | `train_deberta_4090d.py` |
| **BERT** | **0.90444** | `Bert_for_BagOfWords.ipynb` |
| **Word2Vec** | **0.84532** | `Word2vecPart1-3.ipynb` |

### Kaggle 提交结果

#### 1. DeBERTa-v2-XXLarge + LoRA（最佳性能）
![DeBERTa LoRA 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/submission-3-fixed.png)
- **测试准确率**: 0.95984
- **状态**: 完成（3天前）

#### 2. BERT
![BERT 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/submission-2.png)
- **测试准确率**: 0.90444
- **状态**: 完成（21天前）

#### 3. Word2Vec（基线）
![Word2Vec 结果](https://github.com/FujunhaoFc/Word2Vec/blob/main/images/Bag_of_Words_model.png)
- **测试准确率**: 0.84532
- **状态**: 完成（23天前）

## 📁 仓库结构

```
.
├── Word2vecPart1.ipynb          # Word2Vec 实现 - 第1部分
├── Word2vecPart2.ipynb          # Word2Vec 实现 - 第2部分
├── Word2vecPart3.ipynb          # Word2Vec 实现 - 第3部分
├── Bert_for_BagOfWords.ipynb    # 基于 BERT 的情感分类器
├── train_deberta_4090d.py       # DeBERTa-v2-XXLarge + LoRA 微调
└── imdb_modernbert_unsloth.py   # DeBERTa + Unsloth 优化（开发中）
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

### 阶段四：DeBERTa + Unsloth（开发中）
- **文件**: `imdb_modernbert_unsloth.py`
- **状态**: ⚠️ 正在完善中
- **方法**: 使用 Unsloth 框架进一步优化
- **备注**: 性能指标审核中

## 🔬 技术亮点

### 实现的关键创新：

1. **渐进式复杂度**: 展示了从浅层到深度学习架构的演进过程
2. **参数效率**: 使用 LoRA 在有限 GPU 资源下微调十亿参数模型
3. **性能优化**: 
   - Kaggle P100 GPU 实验
   - 租用 RTX 4090D 进行大模型训练
   - 调试 FP16 溢出和 CUDA 内存问题

## 📈 性能分析

模型演进展示了现代架构的明显优势：

- 从 Word2Vec 到 DeBERTa **绝对提升 11.45%**
- 相对基线**提升 13.39%**
- 每个阶段都展示了递减但有意义的收益
- DeBERTa 的解耦注意力机制对情感细微差别高度有效

## 🛠️ 技术栈

- **框架**: PyTorch, Transformers (Hugging Face)
- **模型**: Word2Vec, BERT, DeBERTa-v2-XXLarge
- **优化方法**: LoRA, QLoRA, Unsloth
- **硬件**: Kaggle P100, RTX 4090D
- **数据集**: IMDB 电影评论（二分类情感分类）

## 💡 关键收获

1. **预训练的重要性**: 预训练 Transformer 显著优于传统词嵌入
2. **模型规模**: 更大的模型（DeBERTa-XXLarge）能捕获更细微的情感模式
3. **高效微调**: LoRA 使得在有限资源下训练大模型成为可能
4. **架构设计**: DeBERTa 的增强注意力机制提供了持续的性能改进


## 📝 数据集

**IMDB 电影评论数据集**
- 二分类情感分类（正面/负面）
- 25,000 条评论用于训练
- 高度极化的评论提供清晰的情感信号


