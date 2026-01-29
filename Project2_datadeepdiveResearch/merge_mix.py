import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# ================= 설정 =================
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ★ 방금 학습 끝난 체크포인트 경로 (User 제공)
ADAPTER_PATH = "./checkpoints/mvsm_epoch10_r64_mix" 

# ★ 합쳐진 모델을 저장할 새로운 경로
SAVE_PATH = "./checkpoints/mvsm_epoch10_r64_mix_merged" 
# ========================================

def merge():
    print(f"Loading Base Model: {BASE_MODEL_ID}")
    # BF16으로 로드 (메모리 절약 & 성능 유지)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    
    print(f"Loading LoRA Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("Merging weights (BFloat16)...")
    # 가중치 병합
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    
    print("Saving Processor...")
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    except:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        
    processor.save_pretrained(SAVE_PATH)
    print(f"✨ Merge Complete! Saved to {SAVE_PATH}")

if __name__ == "__main__":
    merge()