import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import os

# ================= 설정 =================
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# 방금 학습 끝난 Epoch 10 체크포인트 경로
ADAPTER_PATH = "./checkpoints/mvsm_epoch10" 
# 합쳐진 모델을 저장할 새로운 경로
SAVE_PATH = "./checkpoints/mvsm_epoch10_merged" 
# ========================================

def merge():
    print(f"Loading Base Model: {BASE_MODEL_ID}")
    # ★ BF16 강제 (NaN 에러 방지 및 메모리 절약)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    
    print(f"Loading LoRA Adapter (Epoch 10): {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("Merging weights (BFloat16)...")
    # LoRA 가중치를 본체에 영구적으로 흡수시킴
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    
    print("Saving Processor...")
    try:
        # 프로세서 설정 저장 (이미지 처리 설정 등)
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    except:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        
    processor.save_pretrained(SAVE_PATH)
    print(f"✨ Merge Complete! Saved to {SAVE_PATH}")

if __name__ == "__main__":
    merge()