import os
import sys

# ==========================================
# 1. GPU 설정 (3번 GPU 사용)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(f"🖥️ GPU Setting: CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# 2. 모델 로드
# ==========================================
print("🚀 Loading Model on GPU 3...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
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
# 3. 핵심 알고리즘 (Entropy & Adaptive Alpha)
# ==========================================
def calculate_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def get_adaptive_alpha(current_probs, sensitivity=5.0):
    entropy = calculate_entropy(current_probs)
    # Threshold 0.7 기준 Sigmoid 적용
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
# 4. 데이터 수집 (Full Scan)
# ==========================================
VIDEO_PATH = "/nas_data2/seungwoo/dataset/epic_data/EPIC-KITCHENS/P01/videos/P01_01.MP4"
STRIDE = 5
prompt_text = "Where is the sink? Answer with one word: Left, Right, or Center."

# 파일 존재 확인
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Error: 파일이 존재하지 않습니다: {VIDEO_PATH}")
    sys.exit(1)

results_log = []
history_adaptive = None

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0

print(f"🎬 Starting Stress Test on {VIDEO_PATH}")
print(f"   - Total Frames: {total_frames}")
print(f"   - Stride: {STRIDE}")

frame_idx = 0
pbar = tqdm(total=total_frames)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Stride 적용
    if frame_idx % STRIDE != 0:
        frame_idx += 1
        pbar.update(1)
        continue
        
    # 이미지 전처리
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt_text}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    # --- [A] Baseline (Greedy) & [B] Ours (Adaptive) ---
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
        
    logits = outputs.scores[0][0][target_ids_tensor]
    
    # Baseline 처리
    prob_base = F.softmax(logits, dim=-1)
    pred_base_idx = torch.argmax(prob_base).item()
    
    # Ours 처리
    prob_adapt, used_alpha = apply_smoothing(logits, history_adaptive, method="adaptive")
    history_adaptive = prob_adapt
    pred_adapt_idx = torch.argmax(prob_adapt).item()
    
    # --- [C] Baseline (Random Sampling) - "The Chaos Mode" ---
    # 모델의 본질적 불안정성을 보기 위해 Random Sampling 수행
    with torch.no_grad():
        outputs_rand = model.generate(
            **inputs, 
            max_new_tokens=1, 
            do_sample=True,     # 랜덤 샘플링 켜기
            temperature=1.0,    # 1.0 = 표준 확률 분포 따름
            top_k=50,           # 상위 50개 중에서만 샘플링 (이상한 단어 방지)
            output_scores=True, 
            return_dict_in_generate=True
        )
    
    logits_rand = outputs_rand.scores[0][0][target_ids_tensor]
    prob_rand = F.softmax(logits_rand, dim=-1)
    pred_rand_idx = torch.argmax(prob_rand).item() # 랜덤하게 선택된 것

    # 시각화용 확률값 저장 (여기서는 0번 클래스 'Left'의 확률을 추적한다고 가정)
    # 실제로는 Center나 Right가 정답일 수도 있지만, 
    # '확률이 얼마나 흔들리는지'를 보는 것이 목적이므로 하나만 추적해도 충분함.
    prob_target_base = prob_base[0].item()
    prob_target_adapt = prob_adapt[0].item()
    prob_target_rand = prob_rand[0].item()

    results_log.append({
        "frame": frame_idx,
        "time": frame_idx / fps,
        "base_pred_idx": pred_base_idx,
        "adapt_pred_idx": pred_adapt_idx,
        "rand_pred_idx": pred_rand_idx,
        "prob_target_base": prob_target_base,
        "prob_target_adapt": prob_target_adapt,
        "prob_target_rand": prob_target_rand,
        "used_alpha": used_alpha
    })
    
    frame_idx += 1
    pbar.update(1)

cap.release()
pbar.close()

# 데이터 저장
df = pd.DataFrame(results_log)
df.to_csv("stress_test_full_data.csv", index=False)
print("💾 Full data saved to 'stress_test_full_data.csv'")

# ==========================================
# 5. Top-5 Hardest Segment Mining & Visualization
# ==========================================
print("🔍 Mining Top-5 Hardest Segments...")

# Baseline(Greedy)의 예측이 바뀐 지점(Flickering) 계산
df['shifted'] = df['base_pred_idx'].shift(1)
df['flicker'] = (df['base_pred_idx'] != df['shifted']).astype(int)

# 5초 구간(윈도우) 내에서 플리커링이 가장 심한 곳 찾기
window_sec = 5
window_size = int(window_sec * (fps / STRIDE)) # 5초에 해당하는 데이터 포인트 수

# 롤링 윈도우로 플리커링 합계 계산
df['rolling_flicker'] = df['flicker'].rolling(window=window_size).sum()

# Top 5 구간 찾기 (겹치지 않게)
top_segments = []
temp_df = df.copy()

for i in range(5):
    if temp_df['rolling_flicker'].max() == 0: break
    
    max_idx = temp_df['rolling_flicker'].idxmax()
    start_idx = max(0, max_idx - window_size)
    end_idx = max_idx
    
    top_segments.append((start_idx, end_idx))
    
    # 이미 찾은 구간 주변 지우기
    clean_start = max(0, start_idx - window_size)
    clean_end = min(len(temp_df), end_idx + window_size)
    temp_df.loc[clean_start:clean_end, 'rolling_flicker'] = 0

print(f"✅ Found {len(top_segments)} critical segments.")

# 시각화 함수
def plot_segment(segment_df, segment_id):
    plt.figure(figsize=(14, 6))
    
    times = segment_df['time']
    
    # 1. Baseline (Random Sampling): 초록색 점선 (가장 불안정함)
    plt.plot(times, segment_df['prob_target_rand'], color='green', linestyle=':', alpha=0.4, label='Baseline (Random Sampling)')
    
    # 2. Baseline (Greedy): 빨간색 점선 (불안정함)
    plt.plot(times, segment_df['prob_target_base'], color='red', linestyle='--', alpha=0.6, label='Baseline (Greedy)')
    plt.scatter(times, segment_df['prob_target_base'], color='red', s=10, alpha=0.6)
    
    # 3. Ours (Adaptive): 파란색 실선 (안정적임)
    plt.plot(times, segment_df['prob_target_adapt'], color='blue', linewidth=2.5, label='Ours (Adaptive)')
    
    # 4. 방어 기제 작동 순간 (High Alpha) 표시
    high_alpha_mask = segment_df['used_alpha'] > 0.7
    if high_alpha_mask.any():
        plt.scatter(times[high_alpha_mask], segment_df.loc[high_alpha_mask, 'prob_target_adapt'], 
                   color='purple', s=40, marker='*', label='High Alpha (>0.7)', zorder=5)

    # TC Score (Greedy vs Ours) 비교
    def calc_tc(preds):
        if len(preds) < 2: return 0.0
        return sum(1 for i in range(len(preds)-1) if preds[i] == preds[i+1]) / (len(preds)-1)
    
    tc_base = calc_tc(segment_df['base_pred_idx'].tolist())
    tc_ours = calc_tc(segment_df['adapt_pred_idx'].tolist())
    
    plt.title(f"Stress Test Case #{segment_id+1} (Time: {times.iloc[0]:.1f}s ~ {times.iloc[-1]:.1f}s)\nBaseline TC: {tc_base:.3f}  vs  Ours TC: {tc_ours:.3f} (Improvement: +{(tc_ours-tc_base)*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Prediction Confidence (Target: Left)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"stress_test_case_{segment_id+1}.png", dpi=300)
    plt.close()

# 그래프 그리기 실행
for i, (start, end) in enumerate(top_segments):
    segment_data = df.iloc[start:end]
    plot_segment(segment_data, i)

print("🎉 All Done! Check 'stress_test_case_*.png' files.")