# source /nas_data2/seungwoo/2/miniconda3/bin/activate viewspatial
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import os

# ================= 설정 =================
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# 방금 학습 끝난 LoRA 어댑터 경로
ADAPTER_PATH = "./checkpoints/mvsm_lora_64_128"
# 합쳐진 모델을 저장할 경로 (Merged)
SAVE_PATH = "./checkpoints/mvsm_lora_64_128_merged"
# ========================================

def merge():
    print(f"Loading Base Model: {BASE_MODEL_ID}")
    # ★ BF16 강제 (Qwen2.5는 BF16 권장)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging weights (BFloat16)...")
    # LoRA 가중치를 본체에 영구적으로 흡수시킴
    model = model.merge_and_unload()

    print(f"Saving merged model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)

    print("Saving Processor (Chat Template & Image Config)...")
    try:
        # 프로세서 설정도 같이 저장해야 나중에 에러가 안 납니다.
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    except:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    processor.save_pretrained(SAVE_PATH)
    print("✨ Merge Complete! Ready to Evaluate.")

if __name__ == "__main__":
    merge()