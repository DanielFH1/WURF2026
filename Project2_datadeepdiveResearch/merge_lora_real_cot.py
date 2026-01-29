import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os

# ================= 설정 =================
# 1. 학습된 Real CoT LoRA 경로
ADAPTER_DIR = "./checkpoints/mvsm_real_cot_v1"

# 2. 베이스 모델 ID (★ 중요: 학습할 때 썼던 2.5 버전이어야 함)
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 3. 병합된 모델 저장 경로
OUTPUT_DIR = "./checkpoints/mvsm_real_cot_merged"
# ========================================

def merge():
    print(f"🚀 Loading Base Model: {BASE_MODEL_ID} (CPU Mode)...")
    # 메모리 절약을 위해 CPU로 로드
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print(f"🚀 Loading Real CoT LoRA Adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    print("🔄 Merging LoRA into Base Model...")
    model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    
    print("💾 Saving processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("✨ Merge Complete! 이제 평가(Evaluate)를 진행하세요.")

if __name__ == "__main__":
    merge()