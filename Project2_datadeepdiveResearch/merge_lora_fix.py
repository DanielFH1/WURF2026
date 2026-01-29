import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os

# ================= 설정 =================
# 1. 학습된 LoRA 어댑터 경로
ADAPTER_DIR = "./checkpoints/mvsm_aug_flip_v1"

# 2. 베이스 모델 ID
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# 3. 병합된 모델 저장 경로
OUTPUT_DIR = "./checkpoints/mvsm_aug_flip_v1_merged"
# ========================================

def merge():
    print(f"🚀 Loading Base Model: {BASE_MODEL_ID} (Offline Mode)")
    
    # [수정] local_files_only=True 추가 (인터넷 차단, 로컬 캐시 사용)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        local_files_only=True 
    )

    print(f"🚀 Loading LoRA Adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    print("🔄 Merging LoRA into Base Model...")
    model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    
    print("💾 Saving processor...")
    # [수정] Processor도 로컬에서만 찾도록 강제
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, local_files_only=True)
        processor.save_pretrained(OUTPUT_DIR)
    except Exception as e:
        print(f"⚠️ Processor 로드 중 경고: {e}")
        print("   -> 학습된 체크포인트 폴더에서 processor 파일을 복사해옵니다.")
        # 만약 베이스 모델 로드 실패시, 학습된 폴더에서 복사 시도
        processor = AutoProcessor.from_pretrained(ADAPTER_DIR, local_files_only=True)
        processor.save_pretrained(OUTPUT_DIR)
    
    print("✨ Merge Complete! 이제 평가(Evaluate) 돌리셔도 됩니다.")

if __name__ == "__main__":
    merge()