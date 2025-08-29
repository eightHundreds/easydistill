# LoRA 蒸馏训练指南

本指南介绍如何在 EasyDistill 中使用 LoRA（Low-Rank Adaptation）进行高效的知识蒸馏训练。

## 前置要求

安装 PEFT 库：
```bash
pip install peft
```

## LoRA 配置说明

在配置文件中添加 `lora` 部分：

```json
{
  "lora": {
    "enable": true,
    "r": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "dropout": 0.1,
    "bias": "none",
    "save_merged_model": false
  }
}
```

### 参数说明

- **enable**: 是否启用 LoRA 训练（true/false）
- **r**: LoRA 的秩（rank），控制适配器的大小，常用值：8, 16, 32, 64
- **alpha**: LoRA 的缩放参数，通常设为 r 的 2 倍
- **target_modules**: 要应用 LoRA 的模块名称列表
- **dropout**: LoRA 层的 dropout 概率
- **bias**: 偏置项处理方式（"none", "all", "lora_only"）
- **save_merged_model**: 是否保存合并后的完整模型

### 常用目标模块

**Qwen 系列模型**:
```json
["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**LLaMA 系列模型**:
```json
["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**Baichuan 系列模型**:
```json
["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

## 训练配置建议

### LoRA 训练参数调整

由于 LoRA 训练只更新少量参数，建议调整以下训练参数：

1. **学习率**: 可以设置更高的学习率（5e-4 到 1e-3）
2. **批次大小**: 可以增加批次大小，减少显存占用
3. **训练轮数**: 可能需要更多训练轮数来达到收敛

示例配置：
```json
{
  "training": {
    "learning_rate": 5e-4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 5
  }
}
```

## 使用示例

### 1. 基础 LoRA 蒸馏训练

```bash
python -m easydistill.kd.train --config configs/kd_white_box_lora.json
```

### 2. 多教师 LoRA 蒸馏训练

```bash
python -m easydistill.kd.multi_train --config configs/kd_white_box_lora.json
```

## 模型保存和加载

### 保存的文件结构

训练完成后，输出目录包含：
```
result_lora/
├── adapter_config.json    # LoRA 适配器配置
├── adapter_model.bin      # LoRA 适配器权重
├── tokenizer.json         # 分词器文件
├── tokenizer_config.json  # 分词器配置
└── merged_model/          # 合并后的完整模型（如果启用）
    ├── config.json
    └── pytorch_model.bin
```

### 加载 LoRA 模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("student/Qwen/Qwen2.5-0.5B-Instruct/")
tokenizer = AutoTokenizer.from_pretrained("student/Qwen/Qwen2.5-0.5B-Instruct/")

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "result_lora/")

# 推理
inputs = tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 合并模型（可选）

如果想要一个独立的模型文件：

```python
from peft import PeftModel

# 加载模型和适配器
base_model = AutoModelForCausalLM.from_pretrained("student/Qwen/Qwen2.5-0.5B-Instruct/")
lora_model = PeftModel.from_pretrained(base_model, "result_lora/")

# 合并适配器到基础模型
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("merged_model/")
```

## 优势和注意事项

### 优势

1. **显存效率**: 只训练少量参数，显著降低显存需求
2. **训练速度**: 更快的训练速度和更少的计算资源
3. **存储效率**: 适配器文件很小，便于分发和部署
4. **灵活性**: 可以为不同任务训练不同的适配器

### 注意事项

1. **性能权衡**: LoRA 可能在某些任务上性能略低于全参数训练
2. **超参数敏感**: r 和 alpha 的选择对性能影响较大
3. **模块选择**: target_modules 的选择需要根据模型架构调整
4. **收敛性**: 可能需要更多轮次才能达到最佳性能

## 故障排除

### 常见问题

1. **ImportError: No module named 'peft'**
   ```bash
   pip install peft
   ```

2. **CUDA out of memory**
   - 减少 `per_device_train_batch_size`
   - 增加 `gradient_accumulation_steps`
   - 降低 `r` 值

3. **目标模块不存在**
   - 检查模型架构，确认 `target_modules` 中的模块名称正确
   - 可以通过 `model.named_modules()` 查看所有模块名称

4. **训练不收敛**
   - 适当提高学习率
   - 增加训练轮数
   - 调整 r 和 alpha 参数
