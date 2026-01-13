# run_experiment.py
import os
import sys

# ==========================================
# 1. GPU 설정 (가장 중요!)
# ==========================================
# 다른 라이브러리(torch 등)를 임포트하기 전에 설정해야 안전합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
print(f"🖥️ GPU Setting: CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# qwen_vl_utils 라이브러리 체크 및 로드
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("⚠️ 'qwen_vl_utils' 라이브러리가 없습니다. 설치를 시도합니다...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-vl-utils"])
    from qwen_vl_utils import process_vision_info
    print("✅ 'qwen-vl-utils' 설치 및 로드 완료!")

# ==========================================
# 2. 모델 로드 & 설정
# ==========================================
print("🚀 Loading Model on GPU 3...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto" # 위에서 CUDA_VISIBLE_DEVICES=3으로 제한했으므로 자동으로 3번에 로드됨
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    sys.exit(1)

# 타겟 단어 설정
target_words = ["Left", "Right", "Center"]
target_ids = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in target_words]
target_ids_tensor = torch.tensor(target_ids).to("cuda")

# ==========================================
# 3. 핵심 알고리즘 함수들
# ==========================================
def calculate_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def get_adaptive_alpha(current_probs, sensitivity=5.0):
    entropy = calculate_entropy(current_probs)
    adaptive_alpha = torch.sigmoid(sensitivity * (entropy - 0.7)).item()
    return np.clip(adaptive_alpha, 0.1, 0.9)

def apply_smoothing(current_logits, history_probs, method="fixed", alpha=0.6):
    current_probs = F.softmax(current_logits, dim=-1)
    if history_probs is None:
        return current_probs, 0.0
    
    if method == "fixed":
        final_alpha = alpha
    elif method == "adaptive":
        final_alpha = get_adaptive_alpha(current_probs)
        
    smoothed_probs = (1 - final_alpha) * current_probs + final_alpha * history_probs
    return smoothed_probs, final_alpha

# ==========================================
# 4. 실험 실행 루프 (Main Loop)
# ==========================================
VIDEO_PATH = "/nas_data2/seungwoo/dataset/epic_data/EPIC-KITCHENS/P01/videos/P01_01.MP4"
MAX_FRAMES = None # 전체 실행
STRIDE = 5

if not os.path.exists(VIDEO_PATH):
    print(f"❌ Error: 파일이 존재하지 않습니다: {VIDEO_PATH}")
    sys.exit(1)

prompt_text = "Where is the sink? Answer with one word: Left, Right, or Center."
results_log = []

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0

print(f"🎬 Starting Full-Scale Experiment")
print(f"   - Video: {VIDEO_PATH}")
print(f"   - Total Frames: {total_frames}")

# 진행바 설정
pbar = tqdm(total=total_frames)

history_fixed = None
history_adaptive = None
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Stride 적용
    if frame_idx % STRIDE != 0:
        frame_idx += 1
        pbar.update(1)
        continue
        
    # 추론
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt_text}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
        
    logits = outputs.scores[0][0][target_ids_tensor]
    
    # 스무딩 적용
    prob_base = F.softmax(logits, dim=-1)
    pred_base = target_words[torch.argmax(prob_base).item()]
    
    prob_fixed, _ = apply_smoothing(logits, history_fixed, method="fixed", alpha=0.6)
    history_fixed = prob_fixed
    pred_fixed = target_words[torch.argmax(prob_fixed).item()]
    
    prob_adapt, used_alpha = apply_smoothing(logits, history_adaptive, method="adaptive")
    history_adaptive = prob_adapt
    pred_adapt = target_words[torch.argmax(prob_adapt).item()]
    
    # 로깅
    results_log.append({
        "frame": frame_idx,
        "time": frame_idx / fps,
        "base_pred": pred_base,
        "fixed_pred": pred_fixed,
        "adapt_pred": pred_adapt,
        "used_alpha": used_alpha,
        "entropy": calculate_entropy(prob_base).item()
    })
    
    frame_idx += 1
    pbar.update(1)

cap.release()
pbar.close()

# 결과 저장
df = pd.DataFrame(results_log)
save_filename = "experiment_results_scaleup_gpu3.csv"
df.to_csv(save_filename, index=False)
print(f"\n💾 Results successfully saved to '{save_filename}'")