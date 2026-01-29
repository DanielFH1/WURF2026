import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import math

# ================= 설정 =================
BASE_DIR = "/nas_data2/seungwoo/2/ViewSpatial-Bench"
TRAIN_FILE = "data_train_scene_split/train.jsonl"
SAVE_PATH = "vis_aug_sample.png"

# ================= 증강 함수 (동일 로직) =================
def get_augmentations(img):
    ops = [
        ("Original", lambda x: x, False),
        ("H-Flip", lambda x: x.transpose(Image.FLIP_LEFT_RIGHT), True),
        ("Bright Up", lambda x: ImageEnhance.Brightness(x).enhance(1.5), False),
        ("Bright Down", lambda x: ImageEnhance.Brightness(x).enhance(0.7), False),
        ("Contrast Up", lambda x: ImageEnhance.Contrast(x).enhance(1.5), False),
        ("Contrast Down", lambda x: ImageEnhance.Contrast(x).enhance(0.7), False),
        ("Blur", lambda x: x.filter(ImageFilter.GaussianBlur(radius=1.5)), False),
        
        ("Flip+BrightUp", lambda x: ImageEnhance.Brightness(x.transpose(Image.FLIP_LEFT_RIGHT)).enhance(1.5), True),
        ("Flip+BrightDn", lambda x: ImageEnhance.Brightness(x.transpose(Image.FLIP_LEFT_RIGHT)).enhance(0.7), True),
        ("Flip+ContUp", lambda x: ImageEnhance.Contrast(x.transpose(Image.FLIP_LEFT_RIGHT)).enhance(1.5), True),
        ("Flip+ContDn", lambda x: ImageEnhance.Contrast(x.transpose(Image.FLIP_LEFT_RIGHT)).enhance(0.7), True),
        ("Flip+Blur", lambda x: x.filter(ImageFilter.GaussianBlur(radius=1.5)).transpose(Image.FLIP_LEFT_RIGHT), True),
    ]
    
    results = []
    for name, func, is_flip in ops:
        results.append((name, func(img), is_flip))
    return results

def visualize():
    # 1. 샘플 이미지 찾기
    print("🔍 샘플 데이터 로딩 중...")
    file_path = os.path.join(BASE_DIR, TRAIN_FILE)
    
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    sample_item = None
    with open(file_path, 'r') as f:
        # 랜덤하게 하나 뽑기 위해 전체를 읽지 않고 앞부분에서 적당히 스킵
        lines = f.readlines()
        random_line = random.choice(lines[:100]) # 앞쪽 100개 중 하나 랜덤
        sample_item = json.loads(random_line)

    # 이미지 경로 추출
    try:
        img_rel_path = sample_item['messages'][0]['content'][0]['image']
        question = sample_item['messages'][0]['content'][1]['text']
        answer = sample_item['messages'][1]['content'][0]['text']
    except:
        print("⚠️ 데이터 포맷이 예상과 다릅니다.")
        return

    full_img_path = os.path.join(BASE_DIR, img_rel_path)
    print(f"📸 선택된 이미지: {full_img_path}")
    
    if not os.path.exists(full_img_path):
        print("❌ 이미지 파일이 존재하지 않습니다.")
        return

    # 2. 증강 적용
    img = Image.open(full_img_path).convert('RGB')
    aug_results = get_augmentations(img)

    # 3. 시각화 (Grid Plot)
    # 3행 4열 = 12개
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

    for i, (name, aug_img, is_flip) in enumerate(aug_results):
        ax = axes[i // cols, i % cols]
        ax.imshow(aug_img)
        
        # 제목 색상: 반전된 경우 빨간색 강조
        color = 'red' if is_flip else 'black'
        label_text = f"{name}\n(Label Flipped)" if is_flip else name
        
        ax.set_title(label_text, color=color, fontweight='bold', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✨ 시각화 완료! 저장된 파일: {SAVE_PATH}")

if __name__ == "__main__":
    visualize()