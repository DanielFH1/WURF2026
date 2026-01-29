import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os

# ================= 설정 =================
# 1. 학습된 CoT LoRA 경로
ADAPTER_DIR = "./checkpoints/mvsm_cot_v1"

# 2. 베이스 모델 ID
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 3. 병합된 모델 저장 경로
OUTPUT_DIR = "./checkpoints/mvsm_cot_merged"
# ========================================

def merge():
    print(f"🚀 Loading Base Model (Offline Mode)...")
    # CPU로 로드해서 메모리 절약
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )

    print(f"🚀 Loading CoT LoRA Adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    print("🔄 Merging LoRA into Base Model...")
    model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    
    print("💾 Saving processor...")
    try:
        # 베이스 모델의 프로세서 복사
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, local_files_only=True)
        processor.save_pretrained(OUTPUT_DIR)
    except Exception as e:
        print(f"⚠️ Warning: {e}")
        # 실패 시 어댑터 폴더에서 복사 시도
        processor = AutoProcessor.from_pretrained(ADAPTER_DIR, local_files_only=True)
        processor.save_pretrained(OUTPUT_DIR)
    
    print("✨ Merge Complete! 이제 평가(Evaluate)를 준비하세요.")

if __name__ == "__main__":
    merge()