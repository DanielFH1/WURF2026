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

# ================= 설정 (Augmentation 적용) =================
# W&B 프로젝트 이름
os.environ["WANDB_PROJECT"] = "ViewSpatial-DeepDive"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# [변경 1] 결과가 저장될 폴더 이름 변경 (Baseline과 섞이지 않게)
OUTPUT_DIR = "./checkpoints/mvsm_aug_flip_v1"

# [변경 2] 데이터 경로 설정
DATA_DIR = "data_train_scene_split"
# ★ 증강된 데이터 파일 사용
TRAIN_FILE = os.path.join(DATA_DIR, "train_augmented.jsonl") 
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")

# [변경 3] 이미지 루트 경로 (절대 경로)
# 이 경로를 이미지 파일명 앞에 붙여서 로더가 파일을 못 찾는 문제 해결
IMAGE_ROOT = "/nas_data2/seungwoo/2/ViewSpatial-Bench"

# ======================================================

def load_dataset(file_path):
    data = []
    print(f"📂 Loading: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            # ★ 이미지 경로 절대 경로로 변환 (File Not Found 방지)
            if 'messages' in item:
                for msg in item['messages']:
                    if msg['role'] == 'user':
                        for content in msg['content']:
                            if content['type'] == 'image':
                                # "augmented_images/..." 등을 "/nas.../augmented_images/..."로 변환
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
        
        # Processor가 이미지를 로드하고 텐서로 변환
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        labels[labels == image_token_id] = -100
        inputs["labels"] = labels
        return inputs

def train():
    # [변경 4] WandB Run 이름 변경
    wandb.init(name="Augment-Flip-Rank16-Epoch3")
    
    # 1. 모델 로드
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)

    model.enable_input_require_grads()

    # 2. LoRA 설정
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

    # 3. 데이터 준비
    train_dataset = load_dataset(TRAIN_FILE)
    val_dataset = load_dataset(VAL_FILE)

    # 4. 학습 인자
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
    print(f"✨ Augmentation 학습 완료: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()