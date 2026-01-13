import os
import sys

# ==========================================
# 1. GPU 설정 (3번 GPU 사용 - 메모리 부족시 다른 번호로 변경)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-vl-utils"])
    from qwen_vl_utils import process_vision_info

# ==========================================
# 2. 모델 로드
# ==========================================
print("🚀 Loading Model for VSB Test...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# VSB 스타일의 답변 후보군 (상대적 위치) [cite: 104]
target_words = ["Left", "Right", "Front", "Back"] # VSB의 주요 Direction
target_ids = [processor.tokenizer.encode(w, add_special_tokens=False)[0] for w in target_words]
target_ids_tensor = torch.tensor(target_ids).to("cuda")

# ==========================================
# 3. VSB 스타일 실험 설정
# ==========================================
VIDEO_PATH = "/nas_data2/seungwoo/dataset/epic_data/EPIC-KITCHENS/P01/videos/P01_01.MP4"

# [VSB Prompt Template 적용] 
# "Where is the {object1} located compared to the {object2} from the camera's perspective?"
# P01_01 영상 내용을 고려하여 'Sponge'(스펀지)와 'Tap'(수도꼭지) 관계를 봅니다.
# (영상을 보시고 물체 이름은 수정해주세요!)
object1 = "sponge"
object2 = "tap"
prompt_text = f"Where is the {object1} located compared to the {object2} from the camera's perspective? Answer with one word: Left, Right, Front, or Back."

print(f"📝 Prompt: {prompt_text}")

# ==========================================
# 4. 데이터 수집 (5분 분량만 테스트 - 18000 프레임)
# ==========================================
# 전체를 다 돌리기보다, 물체 두 개가 같이 나오는 구간이 중요하므로
# 앞부분 5000 프레임 정도만 빠르게 돌려서 경향성을 보는 것을 추천합니다.
MAX_FRAMES = 5000 
STRIDE = 5

results_log = []
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

pbar = tqdm(total=min(MAX_FRAMES, total_frames))
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= MAX_FRAMES: break
    
    if frame_idx % STRIDE != 0:
        frame_idx += 1
        pbar.update(1) # 전체 진행률 위해 업데이트
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
    probs = F.softmax(logits, dim=-1)
    
    pred_idx = torch.argmax(probs).item()
    pred_word = target_words[pred_idx]
    confidence = probs[pred_idx].item()
    
    results_log.append({
        "frame": frame_idx,
        "time": frame_idx / fps,
        "pred_word": pred_word,
        "confidence": confidence,
        "probs": probs.cpu().numpy().tolist() # 전체 확률 분포 저장
    })
    
    frame_idx += 1
    pbar.update(1)

cap.release()
pbar.close()

# CSV 저장
df = pd.DataFrame(results_log)
df.to_csv("vsb_video_test_results.csv", index=False)
print("💾 Data saved to 'vsb_video_test_results.csv'")

# ==========================================
# 5. 시각화 (VSB Failure Visualization)
# ==========================================
plt.figure(figsize=(15, 6))

# 바코드 스타일로 시각화 (답변이 얼마나 바뀌는지 확인)
# 색상 매핑
color_map = {'Left': 'red', 'Right': 'blue', 'Front': 'green', 'Back': 'orange'}
colors = [color_map.get(w, 'gray') for w in df['pred_word']]

# 산점도로 표현 (시간축 vs 예측 단어)
for word in target_words:
    subset = df[df['pred_word'] == word]
    plt.scatter(subset['time'], [word]*len(subset), c=color_map.get(word), label=word, s=10, alpha=0.6)

plt.plot(df['time'], df['pred_word'], c='gray', alpha=0.2, linestyle=':') # 연결선 (흔들림 강조)

plt.title(f"VSB Task on Video: '{prompt_text}'\n(Allocentric Stability Analysis)", fontsize=14)
plt.xlabel("Time (seconds)")
plt.ylabel("Predicted Relative Position")
plt.grid(True, linestyle='--', alpha=0.3)

plt.savefig("vsb_failure_visualization.png", dpi=300)
print("✅ Visualization saved: 'vsb_failure_visualization.png'")