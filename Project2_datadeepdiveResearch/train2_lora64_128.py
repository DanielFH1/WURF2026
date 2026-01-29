# lora rank와 alpha만 64,128로 바꾸고, epoch는 3 그대로

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType

# 수정 1: 저장 경로 변경 (구별하기 위해)
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./checkpoints/mvsm_lora_64_128"

def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = []
        images = []
        
        for item in batch:
            messages = item['messages']
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)
            texts.append(text)
            images.append(image_inputs)

        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        labels[labels == image_token_id] = -100
        inputs["labels"] = labels
        return inputs

def train():
    print(f"Loading Model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Vision Encoder Freeze
    for param in model.visual.parameters():
        param.requires_grad = False

    # Gradient Checkpointing 활성화 (OOM 방지 필수)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,             # lora rank 
        lora_alpha=128,    # lora alpha 
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading Datasets...")
    train_dataset = load_dataset("data_train/train.jsonl")
    val_dataset = load_dataset("data_train/val.jsonl")

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        
        num_train_epochs=3,          # train_epoch를 10으로 늘림.
        
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        
        load_best_model_at_end=True,    # 추가: 학습 중 가장 성능 좋은 모델 자동 저장
        save_total_limit=2,             # 용량 절약을 위해 체크포인트는 2개만 유지
        
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenDataCollator(processor),
    )

    print("🚀 Training Start!")
    trainer.train()
    
    print("Saving Model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✨ Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()