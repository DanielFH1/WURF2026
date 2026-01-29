import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType

# ================= 설정 =================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# 저장 경로 명확하게 (Scene Split + Epoch 10)
OUTPUT_DIR = "./checkpoints/mvsm_epoch10_r64_mix" 
# ========================================

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

    # [버그 수정] use_reentrant=False 필수!
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # [설정] Rank 64, Alpha 128
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,             
        lora_alpha=128,    
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading Scene-Split Datasets...")
    # [데이터] 새로 만든 Scene Split 데이터 로드
    train_dataset = load_dataset("data_train_scene_split/train.jsonl")
    val_dataset = load_dataset("data_train_scene_split/val.jsonl")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        
        num_train_epochs=10,          
        
        learning_rate=2e-5,
        logging_steps=1,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        
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

    print("Training Start (Final: Epoch 10 with rank64)")
    trainer.train()
    
    print("Saving Best Model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✨ Best Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()