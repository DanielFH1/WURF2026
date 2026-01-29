import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import os

# ================= 설정 =================
# GPU 설정 (필요시 변경)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

BASE_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# 방금 학습한 어댑터 경로
ADAPTER_PATH = "./checkpoints/mvsm_baseline_paper"
# 합쳐서 저장할 경로
SAVE_PATH = "./checkpoints/mvsm_baseline_merged"
# ========================================

def merge():
    print(f"🔄 Merging: {ADAPTER_PATH} -> {SAVE_PATH}")
    
    # Base Model 로드
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # LoRA 합치기
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()
    
    # 저장
    model.save_pretrained(SAVE_PATH)
    
    # Processor 저장
    try:
        processor = AutoProcessor.from_pretrained(BASE_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    except:
        processor = AutoProcessor.from_pretrained(BASE_ID)
    processor.save_pretrained(SAVE_PATH)
    
    print(f"✨ Merge 완료! 저장된 경로: {SAVE_PATH}")

if __name__ == "__main__":
    merge()