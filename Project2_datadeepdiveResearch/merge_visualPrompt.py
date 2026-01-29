import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os

# ================= 설정 =================
# 1. 학습된 Visual Prompt LoRA 경로
ADAPTER_DIR = "./checkpoints/mvsm_visual_prompt_v1"

# 2. 베이스 모델 (이미 다운로드됨)
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 3. 저장할 경로
OUTPUT_DIR = "./checkpoints/mvsm_visual_prompt_merged"
# ========================================

def merge():
    print(f"🚀 Loading Base Model (Offline)...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True
    )

    print(f"🚀 Loading LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    print("🔄 Merging...")
    model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    
    print("💾 Saving processor...")
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, local_files_only=True)
        processor.save_pretrained(OUTPUT_DIR)
    except:
        processor = AutoProcessor.from_pretrained(ADAPTER_DIR, local_files_only=True)
        processor.save_pretrained(OUTPUT_DIR)
    
    print("✨ Merge Complete!")

if __name__ == "__main__":
    merge()