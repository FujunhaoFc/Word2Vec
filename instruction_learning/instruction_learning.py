"""
指令学习实战：SST-2情感分析
使用Qwen2.5-3B模型
"""

# ============================================================================
# 第一步：环境配置（关键！必须在import之前）
# ============================================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/huggingface_cache/datasets'

print("✓ 缓存目录已设置到数据盘: /root/autodl-tmp/huggingface_cache\n")

# ============================================================================
# 导入库
# ============================================================================

import torch
import numpy as np
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

print("="*80)
print("指令学习实战：SST-2情感分析")
print("测试模型：Qwen2.5-3B")
print("="*80)

# ============================================================================
# 环境检查
# ============================================================================

print("\n" + "="*80)
print("环境检查")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU型号: {gpu_name}")
    print(f"GPU内存: {gpu_memory:.2f} GB")
else:
    print("⚠ 警告: 未检测到GPU，将使用CPU（速度会很慢）")

# 检查磁盘空间
import subprocess
try:
    result = subprocess.run(['df', '-h', '/root/autodl-tmp'], 
                          capture_output=True, text=True, check=True)
    print("\n数据盘空间:")
    for line in result.stdout.split('\n')[1:2]:
        if line:
            parts = line.split()
            if len(parts) >= 5:
                print(f"  总空间: {parts[1]}, 已用: {parts[2]}, 可用: {parts[3]}, 使用率: {parts[4]}")
except:
    print("  无法检查磁盘空间")

# ============================================================================
# 数据集加载
# ============================================================================

print("\n" + "="*80)
print("加载SST-2数据集")
print("="*80)

print("\n正在加载数据集...")
# ⚠️ 重要：使用validation集，test集的标签是-1（不公开）
test_dataset = load_dataset("glue", "sst2", split="validation[:150]")  # 150个样本

print(f"测试集大小: {len(test_dataset)}")

# 查看数据样例
print("\n数据样例：")
for i in range(3):
    example = test_dataset[i]
    label_text = "正面" if example['label'] == 1 else "负面"
    print(f"\n样本 {i+1}:")
    print(f"  文本: {example['sentence'][:80]}...")
    print(f"  标签: {label_text}")

# ============================================================================
# 指令模板定义
# ============================================================================

print("\n" + "="*80)
print("定义指令模板")
print("="*80)

# Alpaca格式模板（适合Qwen）
ALPACA_TEMPLATE = """Below is a movie review. Analyze its sentiment and respond with ONLY one number: -1 for negative or 1 for positive.

### Review:
{input_text}

### Sentiment (respond with -1 or 1 only):"""

print("已定义Alpaca格式指令模板")

# ============================================================================
# 输出解析函数（严格版）
# ============================================================================

def parse_sentiment_output(text, verbose=False):
    """
    从模型输出中解析情感标签
    严格确保只返回0或1（整数类型）
    
    Args:
        text: 模型生成的文本
        verbose: 是否打印调试信息
    
    Returns:
        int: 0 (负面) 或 1 (正面)
    """
    if text is None or text == "":
        if verbose:
            print(f"  ⚠ 空输出，默认返回正面(1)")
        return 1
    
    original_text = str(text)
    text = original_text.strip().lower()
    
    # 去除可能的markdown代码块标记
    text = text.replace('```', '').replace('`', '').strip()
    
    # 优先级1: 明确查找 -1（负面）
    if '-1' in text or text.startswith('-1') or text.endswith('-1'):
        return 0
    
    # 优先级2: 查找单独的 1（正面）
    if text == '1' or text.startswith('1') or ' 1 ' in text or text.endswith(' 1'):
        return 1
    
    # 优先级3: 关键词匹配（更严格）
    negative_keywords = ['negative', 'bad', 'poor', 'terrible', '-1']
    positive_keywords = ['positive', 'good', 'great', 'excellent', '1']
    
    has_negative = any(kw in text for kw in negative_keywords)
    has_positive = any(kw in text for kw in positive_keywords)
    
    if has_negative and not has_positive:
        return 0
    elif has_positive and not has_negative:
        return 1
    
    # 默认返回1（正面）
    if verbose:
        print(f"  ⚠ 无法明确解析，默认返回正面(1)")
        print(f"    原文: {original_text[:80]}")
    
    return 1

# 测试解析函数
print("\n测试输出解析函数:")
test_cases = [
    ("1", 1),
    ("-1", 0),
    ("The sentiment is positive", 1),
    ("Negative: -1", 0),
    ("positive", 1),
    ("negative", 0),
    ("  1  ", 1),
    ("  -1  ", 0),
]

correct = 0
for output, expected in test_cases:
    result = parse_sentiment_output(output)
    # 确保返回值是整数类型
    assert isinstance(result, int), f"返回值必须是整数，但得到 {type(result)}"
    assert result in [0, 1], f"返回值必须是0或1，但得到 {result}"
    
    status = "✓" if result == expected else "✗"
    correct += (result == expected)
    print(f"  {status} '{output}' -> {result} (期望: {expected})")

print(f"通过率: {correct}/{len(test_cases)}")

# ============================================================================
# 模型配置
# ============================================================================

print("\n" + "="*80)
print("模型配置")
print("="*80)

# 只使用Qwen模型
MODEL_CONFIG = {
    "name": "qwen2.5-3b",
    "path": "Qwen/Qwen2.5-3B-Instruct",
    "max_tokens": 30,
    "template": "alpaca",
    "description": "3B参数，阿里巴巴，中英文能力好",
    "size_gb": 6
}

print(f"\n测试模型: {MODEL_CONFIG['name']}")
print(f"路径: {MODEL_CONFIG['path']}")
print(f"说明: {MODEL_CONFIG['description']}")
print(f"模型大小: ~{MODEL_CONFIG['size_gb']}GB")

# ============================================================================
# 指令学习推理函数
# ============================================================================

def instruction_learning_inference(
    config,
    test_data,
    max_samples=None,
    save_predictions=True
):
    """
    使用指令学习进行情感分析推理
    
    Args:
        config: 模型配置
        test_data: 测试数据集
        max_samples: 最大测试样本数
        save_predictions: 是否保存预测结果
    
    Returns:
        results: 包含准确率、预测和标签的字典
    """
    print(f"\n{'='*80}")
    print(f"开始测试: {config['name']}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 加载模型
        print("\n正在加载模型...")
        print("  提示：如果模型已缓存，加载会很快\n")
        
        load_start = time.time()
        
        pipe = pipeline(
            "text-generation",
            model=config['path'],
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        load_time = time.time() - load_start
        print(f"✓ 模型加载完成 (耗时: {load_time:.1f}秒)\n")
        
        # 准备测试数据
        if max_samples:
            n_samples = min(max_samples, len(test_data))
        else:
            n_samples = len(test_data)
        
        print(f"开始推理 (样本数: {n_samples})...")
        
        # 推理
        predictions = []
        true_labels = []
        failed_samples = []
        raw_outputs = []  # 保存原始输出用于调试
        
        inference_start = time.time()
        
        for i, item in enumerate(tqdm(test_data.select(range(n_samples)), desc="推理进度")):
            try:
                # 构建输入
                prompt = ALPACA_TEMPLATE.format(input_text=item['sentence'])
                
                # 生成
                outputs = pipe(
                    prompt,
                    max_new_tokens=config['max_tokens'],
                    do_sample=False,  # 确定性输出
                    temperature=0.1,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
                
                # 提取生成的文本
                response = outputs[0]['generated_text']
                
                # 去除原始prompt，只保留模型生成的部分
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                
                # 解析输出
                pred_label = parse_sentiment_output(response)
                
                # 确保是整数类型的0或1
                assert isinstance(pred_label, int), f"预测标签必须是整数，得到 {type(pred_label)}"
                assert pred_label in [0, 1], f"预测标签必须是0或1，得到 {pred_label}"
                
                predictions.append(pred_label)
                true_labels.append(int(item['label']))  # 确保标签也是整数
                raw_outputs.append(response[:100])
                
                # 每10个样本打印一次详情
                if (i + 1) % 10 == 0 or i < 5:
                    print(f"\n{'─'*60}")
                    print(f"样本 {i+1}:")
                    print(f"  输入: {item['sentence'][:60]}...")
                    print(f"  模型输出: {response[:80]}")
                    pred_text = "正面" if pred_label == 1 else "负面"
                    true_text = "正面" if item['label'] == 1 else "负面"
                    correct = "✓" if pred_label == item['label'] else "✗"
                    print(f"  预测: {pred_text}({pred_label}), 真实: {true_text}({item['label']}) {correct}")
                
            except Exception as e:
                print(f"\n⚠ 样本 {i} 推理失败: {str(e)[:100]}")
                failed_samples.append(i)
                predictions.append(1)  # 默认预测为正面
                true_labels.append(int(item['label']))
                raw_outputs.append("ERROR")
        
        inference_time = time.time() - inference_start
        
        # 转换为numpy数组以确保类型一致
        predictions = np.array(predictions, dtype=int)
        true_labels = np.array(true_labels, dtype=int)
        
        # 验证数据
        print(f"\n{'='*80}")
        print("数据验证")
        print(f"{'='*80}")
        print(f"预测值范围: {predictions.min()} - {predictions.max()}")
        print(f"真实值范围: {true_labels.min()} - {true_labels.max()}")
        print(f"预测值类别: {np.unique(predictions)}")
        print(f"真实值类别: {np.unique(true_labels)}")
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # 详细分类报告
        print(f"\n{'='*80}")
        print("评估结果")
        print(f"{'='*80}")
        print(f"\n测试样本数: {n_samples}")
        print(f"失败样本数: {len(failed_samples)}")
        print(f"推理总耗时: {inference_time:.1f}秒")
        print(f"平均每样本: {inference_time/n_samples:.2f}秒")
        print(f"\n准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n混淆矩阵:")
        print("       预测负面  预测正面")
        print(f"真实负面  {conf_matrix[0][0]:4d}     {conf_matrix[0][1]:4d}")
        print(f"真实正面  {conf_matrix[1][0]:4d}     {conf_matrix[1][1]:4d}")
        
        print("\n详细分类报告:")
        try:
            report = classification_report(
                true_labels,
                predictions,
                target_names=['负面', '正面'],
                labels=[0, 1],  # 明确指定标签
                digits=4
            )
            print(report)
        except Exception as e:
            print(f"⚠ 生成分类报告失败: {e}")
            # 手动计算
            neg_correct = np.sum((predictions == 0) & (true_labels == 0))
            pos_correct = np.sum((predictions == 1) & (true_labels == 1))
            neg_total = np.sum(true_labels == 0)
            pos_total = np.sum(true_labels == 1)
            print(f"负面准确率: {neg_correct}/{neg_total} = {neg_correct/neg_total*100:.2f}%")
            print(f"正面准确率: {pos_correct}/{pos_total} = {pos_correct/pos_total*100:.2f}%")
        
        # 保存预测结果
        if save_predictions:
            results_data = {
                "model": config['name'],
                "model_path": config['path'],
                "template": config['template'],
                "n_samples": int(n_samples),
                "accuracy": float(accuracy),
                "inference_time": float(inference_time),
                "failed_samples": failed_samples,
                "confusion_matrix": conf_matrix.tolist(),
                "predictions_summary": {
                    "correct": int(np.sum(predictions == true_labels)),
                    "total": int(len(predictions)),
                    "negative_pred": int(np.sum(predictions == 0)),
                    "positive_pred": int(np.sum(predictions == 1)),
                    "negative_true": int(np.sum(true_labels == 0)),
                    "positive_true": int(np.sum(true_labels == 1)),
                },
                "sample_predictions": [
                    {
                        "index": i,
                        "text": test_data[i]['sentence'][:100],
                        "true_label": int(true_labels[i]),
                        "pred_label": int(predictions[i]),
                        "raw_output": raw_outputs[i],
                        "correct": bool(predictions[i] == true_labels[i])
                    }
                    for i in range(min(30, len(predictions)))  # 保存前30个样本
                ]
            }
            
            filename = f"predictions_{config['name']}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print(f"\n预测结果已保存到: {filename}")
        
        # 清理GPU内存
        del pipe
        torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        print(f"\n总耗时: {total_time:.1f}秒")
        print(f"{'='*80}\n")
        
        return {
            "model": config['name'],
            "accuracy": float(accuracy),
            "n_samples": int(n_samples),
            "inference_time": float(inference_time),
            "predictions": predictions.tolist(),
            "true_labels": true_labels.tolist(),
            "confusion_matrix": conf_matrix.tolist()
        }
        
    except Exception as e:
        print(f"\n✗ 模型测试失败: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return None

# ============================================================================
# 执行测试
# ============================================================================

print("\n" + "="*80)
print("开始测试")
print("="*80)

result = instruction_learning_inference(
    config=MODEL_CONFIG,
    test_data=test_dataset,
    max_samples=150,  # 测试150个样本
    save_predictions=True
)

# ============================================================================
# 结果汇总
# ============================================================================

print("\n" + "="*80)
print("实验结果汇总")
print("="*80)

if result is not None:
    print(f"\n✓ 测试成功！")
    print(f"  模型: {result['model']}")
    print(f"  准确率: {result['accuracy']*100:.2f}%")
    print(f"  样本数: {result['n_samples']}")
    print(f"  推理时间: {result['inference_time']:.1f}秒")
    
    # 保存完整结果
    summary = {
        "task": "SST-2 Sentiment Analysis using Instruction Learning",
        "model": MODEL_CONFIG['name'],
        "test_samples": len(test_dataset),
        "result": {
            "accuracy": result['accuracy'],
            "n_samples": result['n_samples'],
            "inference_time": result['inference_time'],
            "confusion_matrix": result['confusion_matrix']
        }
    }
    
    with open("instruction_learning_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n完整结果已保存到: instruction_learning_results.json")
    
    # 创建Markdown报告
    report = f"""# SST-2情感分析 - 指令学习实验报告

## 实验配置

- **任务**: 情感分析（二分类）
- **数据集**: SST-2 (Stanford Sentiment Treebank)
- **测试样本**: {len(test_dataset)}
- **GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
- **模型**: {MODEL_CONFIG['name']}

## 实验结果

| 指标 | 值 |
|------|-----|
| 准确率 | {result['accuracy']*100:.2f}% |
| 样本数 | {result['n_samples']} |
| 推理时间 | {result['inference_time']:.1f}秒 |
| 平均耗时 | {result['inference_time']/result['n_samples']:.2f}秒/样本 |

## 混淆矩阵

```
            预测负面  预测正面
真实负面      {result['confusion_matrix'][0][0]}       {result['confusion_matrix'][0][1]}
真实正面      {result['confusion_matrix'][1][0]}       {result['confusion_matrix'][1][1]}
```

## 关键发现

1. **模型性能**: Qwen2.5-3B在零样本情感分析任务上达到 {result['accuracy']*100:.2f}% 的准确率
2. **推理效率**: 平均每个样本需要 {result['inference_time']/result['n_samples']:.2f} 秒

## 结论

本实验使用Qwen2.5-3B（3B参数）进行零样本情感分析，无需任何训练即可完成任务。
实验证明，精心设计的指令模板可以有效引导小模型完成特定的分类任务。

## 技术细节

- **指令模板**: Alpaca格式
- **推理方式**: 零样本（Zero-shot）
- **输出解析**: 基于规则的严格解析（确保只返回0或1）
- **环境**: 数据盘缓存，优化磁盘使用
- **精度**: FP16半精度推理
"""
    
    with open("experiment_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("实验报告已保存到: experiment_report.md")

else:
    print("\n✗ 测试失败")
    print("  请检查错误信息并重试")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "="*80)
print("实验完成！")
print("="*80)

print("\n生成的文件：")
print("  1. instruction_learning_results.json - 完整实验结果")
print("  2. experiment_report.md - 实验报告")
print("  3. predictions_qwen2.5-3b.json - 详细预测结果")

print("\n后续建议：")
print("  1. 查看 experiment_report.md 了解详细结果")
print("  2. 分析 predictions_*.json 中的错误案例")
print("  3. 尝试优化提示词以提升性能")
print("  4. 可以测试更多样本或其他数据集")

print("\n感谢使用！ 🚀")
print("="*80)