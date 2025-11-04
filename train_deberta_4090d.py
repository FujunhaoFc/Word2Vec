"""
DeBERTa-v2-XXLarge Lora Fine-tuning for IMDB datasets
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import gc
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'train_path': 'labeledTrainData.tsv',
    'test_path': 'testData.tsv',
    'model_name': 'microsoft/deberta-v2-xxlarge',
    'max_length': 256, 
    
    # Training
    'train_batch_size': 2,  
    'eval_batch_size': 4,
    'gradient_accumulation_steps': 8,  
    
    # Optimization
    'learning_rate': 2e-4,  
    'num_epochs': 3,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,  
    'max_grad_norm': 1.0,
    
    # LoRA
    'lora_r': 8,  
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    
    # System
    'gradient_checkpointing': True,
    'fp16': True,
    'dataloader_num_workers': 0,
    'save_strategy': 'epoch',
    'evaluation_strategy': 'epoch',
    'logging_steps': 10,
    'save_total_limit': 1,
    'output_dir': './deberta-imdb-final',
}

print("="*70)
print("DeBERTa-v2-XXLarge - FULLY FIXED VERSION")
print("="*70)

# ============================================================================
# GPU CHECK
# ============================================================================
print("\nGPU CHECK:")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLOADING DATA:")

train_df = pd.read_csv(CONFIG['train_path'], delimiter='\t', quoting=3)
test_df = pd.read_csv(CONFIG['test_path'], delimiter='\t', quoting=3)

print(f"✓ Original train: {train_df.shape}")
print(f"✓ Test: {test_df.shape}")

# VERIFY LABELS
print(f"\nLabel distribution:")
print(train_df['sentiment'].value_counts())
print(f"Labels are: {train_df['sentiment'].unique()}")

# Sample 10000
np.random.seed(42)
train_df = train_df.sample(n=10000, random_state=42).reset_index(drop=True)
print(f"✓ Sampled to: {len(train_df)} samples")
print(f"✓ After sampling - Label dist: {train_df['sentiment'].value_counts().to_dict()}")

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).replace('<br />', ' ').replace('<br>', ' ')
    return ' '.join(text.split())

train_df['review'] = train_df['review'].apply(clean_text)
test_df['review'] = test_df['review'].apply(clean_text)

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['review'].values,
    train_df['sentiment'].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df['sentiment'].values
)

print(f"✓ Train: {len(train_texts)} samples")
print(f"✓ Val: {len(val_texts)} samples")
print(f"✓ Train labels: {np.bincount(train_labels)}")
print(f"✓ Val labels: {np.bincount(val_labels)}")

# ============================================================================
# TOKENIZER
# ============================================================================
print("\nLOADING TOKENIZER:")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
print("✓ Tokenizer loaded")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=CONFIG['max_length'],
        padding=False,
    )

train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
test_dataset = Dataset.from_dict({'text': test_df['review'].values})

print("Tokenizing...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
print("✓ Tokenization complete")

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\nLOADING MODEL:")
torch.cuda.empty_cache()
gc.collect()

# Load in FP32, let Trainer handle FP16
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=2,
    trust_remote_code=True,
).cuda()

print("✓ Model loaded")

if CONFIG['gradient_checkpointing']:
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")

print(f"✓ GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# ============================================================================
# APPLY LORA
# ============================================================================
print("\nAPPLYING LORA:")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=["query_proj", "value_proj"],  
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✓ LoRA applied")

# ============================================================================
# TRAINING ARGS
# ============================================================================
print("\nCONFIGURING TRAINING:")

training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['num_epochs'],
    per_device_train_batch_size=CONFIG['train_batch_size'],
    per_device_eval_batch_size=CONFIG['eval_batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
    
    # Optimization 
    optim="adamw_torch",
    learning_rate=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    max_grad_norm=CONFIG['max_grad_norm'],
    warmup_ratio=CONFIG['warmup_ratio'],  # Use ratio!
    lr_scheduler_type='cosine',  # Cosine decay
    
    # Evaluation
    eval_strategy=CONFIG['evaluation_strategy'],
    save_strategy=CONFIG['save_strategy'],
    save_total_limit=CONFIG['save_total_limit'],
    load_best_model_at_end=True,
    metric_for_best_model='eval_roc_auc',
    greater_is_better=True,
    
    # Performance
    fp16=CONFIG['fp16'],
    dataloader_num_workers=CONFIG['dataloader_num_workers'],
    gradient_checkpointing=CONFIG['gradient_checkpointing'],
    
    # Logging
    logging_steps=CONFIG['logging_steps'],
    report_to='none',
    seed=42,
    disable_tqdm=False,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=-1)[:, 1].numpy()
    roc_auc = roc_auc_score(labels, probs)
    return {'accuracy': accuracy, 'roc_auc': roc_auc}

# ============================================================================
# TRAINER
# ============================================================================
print("\nINITIALIZING TRAINER:")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("✓ Trainer initialized")
print(f"  Effective batch: {CONFIG['train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"  Total steps: {len(train_dataset) // (CONFIG['train_batch_size'] * CONFIG['gradient_accumulation_steps']) * CONFIG['num_epochs']}")

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print("Expected time: 1.5-2 hours")
print("="*70 + "\n")

# Train
train_result = trainer.train()

print("\n" + "="*70)
print("TRAINING COMPLETED")
print("="*70)
print(f"Loss: {train_result.training_loss:.4f}")
print(f"Time: {train_result.metrics['train_runtime']/3600:.2f} hours")

# ============================================================================
# EVALUATE
# ============================================================================
print("\nFINAL EVALUATION:")
eval_results = trainer.evaluate()
print(f"✓ Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"✓ ROC-AUC: {eval_results['eval_roc_auc']:.4f}")

# ============================================================================
# PREDICT
# ============================================================================
print("\nGENERATING PREDICTIONS:")
predictions = trainer.predict(test_dataset)
probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()

submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': probs
})

submission.to_csv('submission.csv', index=False)
print(f"✓ Saved: submission.csv")
print(f"  Shape: {submission.shape}")
print(f"  Range: [{submission['sentiment'].min():.4f}, {submission['sentiment'].max():.4f}]")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"✓ Final ROC-AUC: {eval_results['eval_roc_auc']:.4f}")
print("="*70)