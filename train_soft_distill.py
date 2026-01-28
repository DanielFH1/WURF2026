import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import wandb
from transformers import (
    Qwen2_5_VLForConditionalGeneration, # [학생용] 2.5 버전 클래스
    Qwen2VLForConditionalGeneration,    # [선생님용] 2.0 버전 클래스
    AutoProcessor,
    Trainer,
    TrainingArguments
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType

# ================= 설정 =================
os.environ["WANDB_PROJECT"] = "ViewSpatial-DeepDive"

TEACHER_ID = "Qwen/Qwen2-VL-7B-Instruct"   
STUDENT_ID = "Qwen/Qwen2.5-VL-3B-Instruct" 

OUTPUT_DIR = "./checkpoints/mvsm_soft_distill_final" # 경로 이름 살짝 변경 (안전하게)
DATA_DIR = "data_train_scene_split"
TRAIN_FILE = os.path.join(DATA_DIR, "train_real_cot.jsonl") 
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")
IMAGE_ROOT = "/nas_data2/seungwoo/2/ViewSpatial-Bench"

TEMPERATURE = 2.0  
ALPHA = 0.5       
# =======================================

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
                                if not content['image'].startswith("/"):
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

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        if self.teacher:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_student = model(**inputs)
        loss_ce = outputs_student.loss
        
        with torch.no_grad():
            teacher_device = self.teacher.device
            teacher_inputs = {k: v.to(teacher_device) for k, v in inputs.items() if k != 'labels'}
            outputs_teacher = self.teacher(**teacher_inputs)
        
        logits_student = outputs_student.logits
        logits_teacher = outputs_teacher.logits
        
        # Vocab Size 매칭 (Qwen2 vs Qwen2.5)
        if logits_student.shape[-1] != logits_teacher.shape[-1]:
            diff = logits_student.shape[-1] - logits_teacher.shape[-1]
            if diff > 0:
                pad = torch.full((logits_teacher.shape[0], logits_teacher.shape[1], diff), float('-inf'), device=logits_teacher.device)
                logits_teacher = torch.cat([logits_teacher, pad], dim=-1)
            else:
                logits_teacher = logits_teacher[..., :logits_student.shape[-1]]

        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = loss_fct(
            F.log_softmax(logits_student / self.temperature, dim=-1),
            F.softmax(logits_teacher / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        loss = (self.alpha * loss_kd) + ((1 - self.alpha) * loss_ce)
        
        return (loss, outputs_student) if return_outputs else loss

def train():
    wandb.init(name="Soft-Distill-Final-GPU") 
    
    # 1. Student (Qwen2.5) 로드
    print("🚀 Loading Student Model (Qwen2.5-3B)...")
    student_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(STUDENT_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    student_model.enable_input_require_grads()
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    student_model = get_peft_model(student_model, peft_config)
    student_model.print_trainable_parameters()

    # 2. Teacher (Qwen2) 로드 -> Qwen2VL 클래스 사용!
    print("🚀 Loading Teacher Model (Qwen2-7B)...")
    teacher_model = Qwen2VLForConditionalGeneration.from_pretrained(
        TEACHER_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    teacher_model.eval()

    train_dataset = load_dataset(TRAIN_FILE)
    val_dataset = load_dataset(VAL_FILE)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16,
        num_train_epochs=3,         
        learning_rate=2e-5,         
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to="wandb",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        overwrite_output_dir=True
    )

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=TEMPERATURE,
        alpha=ALPHA,
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=QwenDataCollator(processor)
    )
    
    student_model.config.use_cache = False 
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['base_model_name_or_path'] = STUDENT_ID
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✨ Soft Target Distillation 완료: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()