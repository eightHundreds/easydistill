
# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import argparse
import logging
import os
from jinja2 import Environment, BaseLoader, FileSystemLoader
from datasets import load_dataset,Dataset
from typing import Optional, Dict, Union, List
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase,AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer,SFTConfig
import torch
import jsonlines
import numpy as np
import torch.nn.functional as F

# PEFT 相关导入
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    logging.warning("PEFT not available. LoRA training will be disabled.")
    PEFT_AVAILABLE = False


def setup_lora_model(model, lora_config):
    """设置 LoRA 模型
    
    Args:
        model: 预训练模型
        lora_config: LoRA 配置字典
        
    Returns:
        配置了 LoRA 的模型
    """
    if not PEFT_AVAILABLE:
        logging.error("PEFT is not available. Please install peft library: pip install peft")
        raise ImportError("PEFT library is required for LoRA training")
    
    if not lora_config.get("enable", False):
        logging.info("LoRA is disabled in config")
        return model
    
    # 创建 LoRA 配置
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        lora_dropout=lora_config.get("dropout", 0.1),
        bias=lora_config.get("bias", "none"),
        task_type="CAUSAL_LM"
    )
    
    # 应用 LoRA 到模型
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数统计
    model.print_trainable_parameters()
    
    logging.info(f"LoRA configured with r={peft_config.r}, alpha={peft_config.lora_alpha}")
    logging.info(f"Target modules: {peft_config.target_modules}")
    
    return model


class DistillSFTTrainer(SFTTrainer):

    def __init__(
        self,
        logits_dir: str = None,  
        teacher_vocab_size = None,  
        kd_ratio: float = 0.5,    
        max_seq_length : int = 1024,
        distillation_type: str = "forward_kld",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logits_dir = logits_dir
        self.teacher_vocab_size = teacher_vocab_size
        self.kd_ratio = kd_ratio
        self.max_seq_length = max_seq_length
        self.distillation_type = distillation_type
        self.teacher_logits = []
        with jsonlines.open(self.logits_dir) as reader:
            for obj in reader:
                self.teacher_logits.append(obj)


    def _load_teacher_logits(self, batch_size: int, it: int, dp_rank: int, device: torch.device, no_model_batch: Dict):
        start_idx = dp_rank * batch_size + batch_size * it
        end_idx = dp_rank * batch_size + batch_size * (it + 1)
        loaded_data = self.teacher_logits[start_idx:end_idx]
        arr = np.zeros((batch_size, self.max_seq_length, self.teacher_vocab_size))
        for i in range(len(loaded_data)):
            for j in range(len(loaded_data[i])):
                keys = np.array(list(loaded_data[i][j].keys()), dtype=int)
                values = np.array(list(loaded_data[i][j].values()))
                arr[i, j, keys] = values
                
        logits_tensor = torch.tensor(arr, dtype=torch.bfloat16, device=device)
        return self._shift_tensor_right(logits_tensor, no_model_batch['label'], pad_value=0)
    

    def _compute_white_box_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: Optional[torch.Tensor]):
        """计算白盒蒸馏损失
        
        Args:
            student_logits: 学生模型的logits张量，形状为(batch_size, seq_len, vocab_size)
            teacher_logits: 教师模型的logits张量，已转换为概率分布
            labels: 标签张量，用于生成掩码，-100表示需要忽略的位置
            
        Returns:
            torch.Tensor: 计算得到的蒸馏损失
        """
        # 截断学生logits到指定的最大序列长度
        student_logits = student_logits[:, :self.max_seq_length, :]
        
        # 对齐教师概率分布的维度，确保与学生logits形状匹配
        teacher_probs = teacher_logits[:, :student_logits.size(1), :student_logits.size(-1)]
        
        # 创建掩码：标签不为-100的位置为1，否则为0
        # 掩码用于只在有效token位置计算损失
        mask = (labels != -100).float() if labels is not None else torch.ones_like(student_logits[:, :, 0])
        
        if self.distillation_type == "forward_kld":
            # 前向KL散度：学生学习教师的分布
            # KL(teacher || student) = sum(teacher * log(teacher/student))
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),  # 学生分布的对数概率
                teacher_probs,  # 教师分布的概率
                reduction='none',  # 不进行维度约减
                log_target=False  # teacher_probs不是对数形式
            ).sum(dim=-1) / torch.sum(mask.view(-1), dim=0)  # 按词汇维度求和，并归一化
        elif self.distillation_type == "reverse_kld":
            # 反向KL散度：教师为学生提供确定性指导
            # KL(student || teacher) = sum(student * log(student/teacher))
            loss = F.kl_div(
                torch.log(teacher_probs.clamp(min=1e-10)),  # 教师分布的对数概率，避免log(0)
                F.softmax(student_logits, dim=-1),  # 学生分布的概率
                reduction='none',  # 不进行维度约减
                log_target=False  # 学生概率不是对数形式
            ).sum(dim=-1) / torch.sum(mask.view(-1), dim=0)  # 按词汇维度求和，并归一化
        else:
            raise ValueError(f"不支持的蒸馏类型: {self.distillation_type}. 请使用 'forward_kld' 或 'reverse_kld'")
            
        # 应用掩码并计算平均损失，只在有效token位置计算损失
        return (loss * mask).sum() / mask.sum()


    @staticmethod
    def _shift_tensor_right(inputs: torch.Tensor, labels: torch.Tensor, pad_value: float = 0.0):
        batch_size, seqlen, vocab_size = inputs.shape
        device = inputs.device
        labels_ne = labels != -100
        shift_distances = torch.argmax(labels_ne.int(), dim=1)
        idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(batch_size, seqlen)
        shifted_idx = idx - shift_distances.unsqueeze(1)
        mask = shifted_idx >= 0
        shifted_idx = shifted_idx.clamp(min=0)
        inputs_flat = inputs.view(batch_size, seqlen, vocab_size)
        shifted_idx = shifted_idx.unsqueeze(2).expand(-1, -1, vocab_size)
        gathered = torch.gather(inputs_flat, 1, shifted_idx)
        mask = mask.unsqueeze(2).expand(-1, -1, vocab_size)
        return torch.where(mask, gathered, torch.full_like(gathered, pad_value))


    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):
        """计算训练损失，包括语言模型损失和蒸馏损失
        
        Args:
            model: 学生模型
            inputs: 输入数据字典，包含input_ids和labels
            return_outputs: 是否返回模型输出
            num_items_in_batch: 批次中的项目数量
            
        Returns:
            total_loss: 总损失
            outputs: 模型输出（如果return_outputs=True）
        """
        # 通过学生模型前向传播获取输出和语言模型损失
        outputs = model(**inputs)
        lm_loss = outputs.loss
        
        # 如果指定了教师模型logits目录，进行白盒蒸馏
        if self.logits_dir:
            # 加载对应的教师模型logits数据
            teacher_logits = self._load_teacher_logits(
                batch_size=inputs['input_ids'].size(0),  # 当前批次大小
                it=self.state.global_step,  # 当前训练步数
                dp_rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,  # 数据并行rank
                device=model.device,  # 设备
                no_model_batch={'label': inputs.get('labels', None)}  # 标签信息用于对齐
            )
            
            # 计算白盒蒸馏损失（学生logits与教师logits的KL散度）
            distil_loss = self._compute_white_box_distillation_loss(
                student_logits=outputs.logits,  # 学生模型的logits
                teacher_logits=teacher_logits,  # 教师模型的logits
                labels=inputs.get('labels', None)  # 用于创建掩码，只在输出token位置计算损失
            )
            
            # 组合损失：(1-kd_ratio)*语言模型损失 + kd_ratio*蒸馏损失
            total_loss = (1 - self.kd_ratio) * lm_loss + self.kd_ratio * distil_loss
        else:
            # 没有教师模型时，只使用语言模型损失
            total_loss = lm_loss
            
        # 根据return_outputs参数决定返回格式
        return (total_loss, outputs) if return_outputs else total_loss


def formatting_func(examples):
    env = Environment(loader=BaseLoader())
    try:
        message = {"content": examples["instruction"],"output":examples["output"]}
        # 从全局配置中获取system_prompt，如果没有则使用默认值
        system_prompt = global_config.get("dataset", {}).get("system_prompt", "You are a helpful assistant.")
        full_text = template.render(
            message=message,
            system_prompt=system_prompt,
            add_generation_prompt=False,
            add_output=True
        )
        return full_text
    except Exception as e:
        logging.warning(f"Error processing sample: {str(e)}")
        return ""


def train(config):
    dataset = load_dataset("json", data_files=config["dataset"]["labeled_path"])
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True
    )
    
    # 设置 LoRA（如果配置中启用）
    if "lora" in config:
        student_model = setup_lora_model(student_model, config["lora"])

    global template, global_config
    global_config = config  # 使formatting_func能够访问配置
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    training_arguments = SFTConfig(**config["training"])
    
    try:
        job_type =  config["job_type"]
        if "kd_black_box" in job_type:
            dataset = dataset.shuffle(seed=config["dataset"]["seed"])
            trainer = SFTTrainer(
                model=student_model,
                processing_class=student_tokenizer,
                args=training_arguments,
                train_dataset=dataset["train"],
                formatting_func=formatting_func
            )
        elif "kd_white_box" in job_type:
            teacher_vocab_size=json.load(open(os.path.join(config["models"]["teacher"], 'config.json')))['vocab_size']
            trainer = DistillSFTTrainer(
                logits_dir=config["dataset"]["logits_path"],
                teacher_vocab_size=teacher_vocab_size,
                kd_ratio=config["distillation"]["kd_ratio"], 
                max_seq_length=config["distillation"]["max_seq_length"],
                distillation_type=config["distillation"].get("distillation_type", "forward_kld"),
                model=student_model,
                processing_class=student_tokenizer,
                args=training_arguments,
                train_dataset=dataset["train"],
                formatting_func=formatting_func
            )
        else:
            logging.error(f"Invalid job type: {job_type}")
            raise ValueError(f"Invalid job type: {job_type}")
    except ValueError as e:
        logging.error(f"Training job terminated: {e}")
        return
        
    trainer.train()
    
    # 保存模型
    output_dir = config["training"]["output_dir"]
    if "lora" in config and config["lora"].get("enable", False):
        # 如果使用 LoRA，保存适配器
        trainer.model.save_pretrained(output_dir)
        logging.info(f"LoRA adapters saved to {output_dir}")
        
        # 也可以选择合并并保存完整模型
        if config["lora"].get("save_merged_model", False):
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(os.path.join(output_dir, "merged_model"))
            logging.info(f"Merged model saved to {os.path.join(output_dir, 'merged_model')}")
    else:
        # 常规全参数模型保存
        trainer.save_model(output_dir)
    
    student_tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)  


if __name__ == "__main__":
    main()