# ==========================================
import os
# [OOM 방지] 메모리 파편화 방지 설정
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

# ================= 설정 (논문 Baseline) =================
# W&B 설정
os.environ["WANDB_PROJECT"] = "ViewSpatial-DeepDive"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./checkpoints/mvsm_baseline_paper"

# ★ 중요: Data Hygiene이 지켜진 Clean Split 데이터 사용
DATA_DIR = "data_train_scene_split"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")

# ======================================================

def load_dataset(file_path):
    data = []
    print(f"📂 Loading: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
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
    wandb.init(name="Baseline-Rank16-Epoch3")
    
    # 1. 모델 로드
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)

    # ★ [핵심 수정 1] Gradient Checkpointing 사용 시 필수 설정
    # 입력 임베딩 레이어가 그라디언트를 계산하도록 강제합니다. (이게 없으면 에러 발생)
    model.enable_input_require_grads()

    # 2. LoRA 설정 (논문 세팅: Rank 16)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,               # 논문 Baseline
        lora_alpha=32,      # 보통 Rank의 2배
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 데이터 준비
    train_dataset = load_dataset(TRAIN_FILE)
    val_dataset = load_dataset(VAL_FILE)

    # 4. 학습 인자
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # ★ [핵심 수정 2] 메모리 안전 설정
        # 배치 1로 줄이고, Accumulation을 16으로 늘려서 학습 효과 유지 + 메모리 절약
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16,
        
        # Gradient Checkpointing 활성화 (메모리 절약)
        gradient_checkpointing=True,
        
        num_train_epochs=3,         # 논문 Baseline
        learning_rate=2e-5,         # Qwen 기본 권장 LR
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

    # [중요] Gradient Checkpointing 사용 시 use_cache=False 강제
    model.config.use_cache = False 

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✨ Baseline 학습 완료: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()