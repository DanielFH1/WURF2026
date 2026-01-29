# ==========================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ==========================================

import json
import torch
import wandb
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType

# ================= 설정 (CoT 실험) =================
os.environ["WANDB_PROJECT"] = "ViewSpatial-DeepDive"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# [변경] 저장 경로
OUTPUT_DIR = "./checkpoints/mvsm_cot_v1"

# [변경] 데이터셋 경로 (CoT 데이터 사용)
DATA_DIR = "data_train_scene_split"
TRAIN_FILE = os.path.join(DATA_DIR, "train_cot.jsonl") 
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl") # 검증은 원본 포맷으로 해도 됨 (Loss만 볼거면)

IMAGE_ROOT = "/nas_data2/seungwoo/2/ViewSpatial-Bench"
# =================================================

def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'messages' in item:
                for msg in item['messages']:
                    if msg['role'] == 'user':
                        for content in msg['content']:
                            if content['type'] == 'image':
                                content['image'] = os.path.join(IMAGE_ROOT, content['image'])
            data.append(item)
    return data

class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        texts, images = [], []
        for item in batch:
            messages = item['messages']
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            texts.append(text)
            images.append(image_inputs)
        
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        labels[labels == image_token_id] = -100
        inputs["labels"] = labels
        return inputs

def train():
    wandb.init(name="Spatial-CoT-Rank16-Epoch3")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,               
        lora_alpha=32,      
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset(TRAIN_FILE)
    val_dataset = load_dataset(VAL_FILE)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        num_train_epochs=3,         
        learning_rate=2e-5,         
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to="wandb",
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenDataCollator(processor)
    )
    model.config.use_cache = False 
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    # 설정 파일 Base Model 수정 (미리 해두기)
    config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['base_model_name_or_path'] = MODEL_ID
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✨ CoT 학습 완료: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()