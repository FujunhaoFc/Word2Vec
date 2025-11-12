"""
IMDB Sentiment Analysis with Unsloth + SCL + R-Drop
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# ================================
# 配置参数
# ================================
class Config:
    # 数据路径
    train_path = "labeledTrainData.tsv"
    test_path = "testData.tsv"
    
    # 模型配置
    model_name = "unsloth/Llama-3.2-3B-Instruct"  # 推荐使用 3B 模型
    max_seq_length = 512
    load_in_4bit = True  # 4-bit 量化，节省显存
    
    # LoRA 配置
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
    
    # 训练参数
    batch_size = 8
    gradient_accumulation_steps = 2  # 有效 batch_size = 8 * 2 = 16
    num_epochs = 3
    learning_rate = 2e-4
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    
    # SCL 和 R-Drop 参数
    scl_temperature = 0.07  # SCL 温度参数
    scl_weight = 0.1  # SCL 损失权重
    rdrop_alpha = 5.0  # R-Drop KL 散度权重
    use_scl = True
    use_rdrop = True
    
    # 其他
    seed = 42
    output_dir = "./imdb_scl_rdrop_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# 设置随机种子
# ================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================
# 数据集类
# ================================
class IMDBDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, is_test=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review = row['review']
        
        # 构造 instruction prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a sentiment analysis expert. Classify the sentiment of the movie review as positive (1) or negative (0).<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        Review: {review}
        
        Sentiment:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        
        """
        
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        if not self.is_test:
            item['labels'] = torch.tensor(row['sentiment'], dtype=torch.long)
            
        return item


# ================================
# SCL 损失函数
# ================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        features: [batch_size, hidden_dim]
        labels: [batch_size]
        """
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # 去除对角线（自己和自己的相似度）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # 计算 log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算对比损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()
        
        return loss


# ================================
# R-Drop KL 散度损失
# ================================
def compute_kl_loss(p_logits, q_logits):
    
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    
    kl_loss = F.kl_div(p, q, reduction='batchmean', log_target=True)
    kl_loss += F.kl_div(q, p, reduction='batchmean', log_target=True)
    
    return kl_loss / 2.0


# ================================
# 模型包装器（支持分类头）
# ================================
class SentimentClassifier(nn.Module):
    def __init__(self, base_model, hidden_size=3072, num_classes=2):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, return_features=False):
        # 获取模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后一层的 [EOS] token 表示
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        
        # 获取每个序列最后一个有效 token 的表示
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        features = hidden_states[torch.arange(batch_size), sequence_lengths]
        
        # 分类 - 确保 dtype 一致
        features = self.dropout(features)
        # 将 classifier 权重转换为与 features 相同的 dtype
        if features.dtype != self.classifier.weight.dtype:
            features = features.to(self.classifier.weight.dtype)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


# ================================
# 训练函数
# ================================
def train_epoch(model, train_loader, optimizer, scheduler, config, scl_criterion):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_scl_loss = 0
    total_kl_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        # 第一次前向传播
        logits1, features1 = model(input_ids, attention_mask, return_features=True)
        ce_loss1 = F.cross_entropy(logits1, labels)
        
        # R-Drop: 第二次前向传播
        if config.use_rdrop:
            logits2, features2 = model(input_ids, attention_mask, return_features=True)
            ce_loss2 = F.cross_entropy(logits2, labels)
            ce_loss = (ce_loss1 + ce_loss2) / 2.0
            
            # KL 散度损失
            kl_loss = compute_kl_loss(logits1, logits2)
        else:
            ce_loss = ce_loss1
            kl_loss = torch.tensor(0.0).to(config.device)
            features2 = features1
        
        # SCL 损失
        if config.use_scl:
            # 合并两次前向传播的特征
            all_features = torch.cat([features1, features2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            scl_loss = scl_criterion(all_features, all_labels)
        else:
            scl_loss = torch.tensor(0.0).to(config.device)
        
        # 总损失
        loss = ce_loss + config.scl_weight * scl_loss + config.rdrop_alpha * kl_loss
        loss = loss / config.gradient_accumulation_steps
        
        loss.backward()
        
        if (progress_bar.n + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * config.gradient_accumulation_steps
        total_ce_loss += ce_loss.item()
        total_scl_loss += scl_loss.item()
        total_kl_loss += kl_loss.item()
        
        preds = torch.argmax(logits1, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_scl_loss = total_scl_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, avg_ce_loss, avg_scl_loss, avg_kl_loss, accuracy


# ================================
# 验证函数
# ================================
@torch.no_grad()
def validate(model, val_loader, config):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


# ================================
# 预测函数
# ================================
@torch.no_grad()
def predict(model, test_loader, config):
    model.eval()
    predictions = []
    
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
    
    return predictions


# ================================
# 主函数
# ================================
def main():
    config = Config()
    set_seed(config.seed)
    
    print("=" * 50)
    print("IMDB Sentiment Analysis with SCL + R-Drop")
    print(f"Model: {config.model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Use SCL: {config.use_scl}, Use R-Drop: {config.use_rdrop}")
    print("=" * 50)
    
    # 加载数据
    print("\n📊 Loading data...")
    train_df = pd.read_csv(config.train_path, sep='\t')
    test_df = pd.read_csv(config.test_path, sep='\t')
    
    # 数据划分 (80% train, 20% val)
    train_df = train_df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    val_size = int(0.2 * len(train_df))
    val_df = train_df[:val_size]
    train_df = train_df[val_size:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 加载模型和 tokenizer
    print("\n🤖 Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # 自动选择
        load_in_4bit=config.load_in_4bit,
    )
    
    # 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    
    # 包装成分类模型
    print("\n🔧 Creating sentiment classifier...")
    sentiment_model = SentimentClassifier(
        base_model=model,
        hidden_size=3072,  # Llama-3.2-3B 的隐藏层大小
        num_classes=2
    ).to(config.device)
    
    # 确保 classifier 使用正确的 dtype (与 base model 一致)
    # 检测 base model 的 dtype
    base_dtype = next(model.parameters()).dtype
    print(f"Base model dtype: {base_dtype}")
    if base_dtype == torch.bfloat16:
        sentiment_model.classifier = sentiment_model.classifier.to(torch.bfloat16)
        print("Classifier converted to bfloat16")
    
    # 创建数据集
    train_dataset = IMDBDataset(train_df, tokenizer, config.max_seq_length)
    val_dataset = IMDBDataset(val_df, tokenizer, config.max_seq_length)
    test_dataset = IMDBDataset(test_df, tokenizer, config.max_seq_length, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(sentiment_model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # SCL 损失
    scl_criterion = SupervisedContrastiveLoss(temperature=config.scl_temperature)
    
    # 训练
    print("\n🚀 Starting training...\n")
    best_val_acc = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*50}")
        
        train_loss, ce_loss, scl_loss, kl_loss, train_acc = train_epoch(
            sentiment_model, train_loader, optimizer, scheduler, config, scl_criterion
        )
        
        val_loss, val_acc = validate(sentiment_model, val_loader, config)
        
        print(f"\n📈 Epoch {epoch + 1} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, CE: {ce_loss:.4f}, SCL: {scl_loss:.4f}, KL: {kl_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"\n💾 New best model! Validation accuracy: {val_acc:.4f}")
            os.makedirs(config.output_dir, exist_ok=True)
            sentiment_model.base_model.save_pretrained(f"{config.output_dir}/best_model")
            torch.save(sentiment_model.classifier.state_dict(), 
                      f"{config.output_dir}/classifier_head.pth")
    
    # 预测测试集
    print("\n🔮 Predicting test set...")
    predictions = predict(sentiment_model, test_loader, config)
    
    # 保存提交文件
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': predictions
    })
    submission.to_csv('submission_scl_rdrop.csv', index=False)
    print("\n✅ Prediction completed! Results saved to 'submission_scl_rdrop.csv'")
    print(f"\n🏆 Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()