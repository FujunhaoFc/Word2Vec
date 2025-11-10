import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import unsloth
import torch
import sys
import logging
import evaluate

import pandas as pd
import numpy as np

from unsloth import FastModel, FastLanguageModel
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, training_args
from datasets import Dataset

from sklearn.model_selection import train_test_split

# 读取数据
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # 数据分割 - 只对训练集进行分割
    train, val = train_test_split(train, test_size=.2)

    # 数据预处理和清理
    def clean_text(text):
        if pd.isna(text):
            return ""
        # 移除HTML标签和多余空格
        text = str(text)
        text = text.replace('<br />', ' ')
        text = ' '.join(text.split())
        return text

    def truncate_text(text, max_length=512):
        words = text.split()
        if len(words) > max_length:
            return ' '.join(words[:max_length])
        return text

    # 应用预处理
    train['review'] = train['review'].apply(clean_text)
    val['review'] = val['review'].apply(clean_text)
    test['review'] = test['review'].apply(clean_text)

    train['review'] = train['review'].apply(lambda x: truncate_text(x, 512))
    val['review'] = val['review'].apply(lambda x: truncate_text(x, 512))
    test['review'] = test['review'].apply(lambda x: truncate_text(x, 512))

    # 创建数据集字典 - 注意：测试集没有label
    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}  # 测试集没有sentiment标签

    # 转换为Dataset对象
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 模型设置
    model_name = "microsoft/deberta-v2-xxlarge"
    NUM_CLASSES = 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
        max_seq_length=512,
        dtype=None,
        auto_model=AutoModelForSequenceClassification,
        num_labels=NUM_CLASSES,
        gpu_memory_utilization=0.6
    )

    # LoRA配置
    model = FastModel.get_peft_model(
        model,
        r=64,  # 增加rank
        lora_alpha=128,
        lora_dropout=0.1,  # 添加dropout
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "query_proj", "value_proj", "key_proj", "output_proj",
            "dense", "classifier"
        ],
        task_type="SEQ_CLS",
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 评估指标
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 分词函数
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            max_length=512, 
            truncation=True, 
            padding="max_length",
            return_tensors=None
        )

    # 应用分词
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=['text']
    )
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=['text']
    )
    test_dataset = test_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=['text']
    )

    # 设置数据格式
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # 训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=training_args.OptimizerNames.ADAMW_TORCH,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        num_train_epochs=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=2,
        group_by_length=False,
        remove_unused_columns=False,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 训练前评估
    print("初始评估:")
    initial_eval = trainer.evaluate()
    print(f"初始验证损失: {initial_eval['eval_loss']:.4f}, 准确率: {initial_eval['eval_accuracy']:.4f}")

    # 训练模型
    print("开始训练...")
    trainer.train()

    # 保存模型
    model.save_pretrained("./deberta_finetuned")
    tokenizer.save_pretrained("./deberta_finetuned")

    # 模型预测
    print("开始预测测试集...")
    
    # 确保模型处于正确模式
    model.eval()

    # 进行预测
    with torch.no_grad():
        prediction_outputs = trainer.predict(test_dataset)

    print(f"预测完成，输出: {prediction_outputs}")

    # 处理预测结果
    test_logits = prediction_outputs.predictions
    test_pred = np.argmax(test_logits, axis=-1).flatten()

    print(f"预测结果形状: {test_pred.shape}")
    print(f"预测分布: {np.bincount(test_pred)}")

    # 保存结果 - 只保存必要的列
    result_output = pd.DataFrame({
        "id": test["id"], 
        "sentiment": test_pred
    })
    result_output.to_csv("deberta_improved_fixed.csv", index=False, quoting=3)


    
    # 总结
    print("\n=== 训练完成总结 ===")
    print(f"训练数据: {len(train_dataset)} 条")
    print(f"验证数据: {len(val_dataset)} 条")
    print(f"测试数据: {len(test_dataset)} 条")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"最终预测结果: {len(test_pred)} 条")
    print(f"正类预测数量: {np.sum(test_pred)}")
    print(f"负类预测数量: {len(test_pred) - np.sum(test_pred)}")
    
    logging.info('训练和预测完成！')