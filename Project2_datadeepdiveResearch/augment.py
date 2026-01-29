import json
import os
import shutil
from PIL import Image
from tqdm import tqdm

# ================= CONFIGURATION =================
# 작업 경로 설정 (사용자 경로 기반)
BASE_DIR = "/nas_data2/seungwoo/2/ViewSpatial-Bench"
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_augmented.jsonl")

# 새로 생성될 증강 이미지가 저장될 폴더
AUG_IMG_DIR = os.path.join(BASE_DIR, "augmented_images")

# 방향 매핑 (대소문자 주의)
FLIP_MAPPING = {
    "Left": "Right",
    "Right": "Left",
    "Front-Left": "Front-Right",
    "Front-Right": "Front-Left",
    "Back-Left": "Back-Right",
    "Back-Right": "Back-Left",
    "left": "right",
    "right": "left",
    # Front, Back은 유지
    "Front": "Front",
    "Back": "Back",
    "front": "front",
    "back": "back"
}
# =================================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def swap_text(text):
    """텍스트(질문/답변) 내의 방향 단어를 반전시킵니다."""
    # 간단한 단어 치환 (복잡한 문장일 경우 정교한 토크나이징 필요할 수 있음)
    words = text.split()
    new_words = []
    for word in words:
        # 구두점 제거 등은 상황에 맞춰 처리 (여기서는 단순 치환)
        clean_word = word.strip(".,?!")
        if clean_word in FLIP_MAPPING:
            # 매핑된 단어로 교체 (원래 구두점 등 유지 필요 시 추가 로직 필요하지만, 보통 라벨은 단어 자체임)
            replaced = FLIP_MAPPING[clean_word]
            new_words.append(word.replace(clean_word, replaced))
        else:
            new_words.append(word)
    return " ".join(new_words)

def process_augmentation():
    print(f"🚀 Starting Horizontal Flip Augmentation...")
    print(f"📂 Reading from: {INPUT_JSONL}")
    
    ensure_dir(AUG_IMG_DIR)
    
    new_entries = []
    
    # 1. 원본 데이터 로드
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📊 Total original samples: {len(lines)}")
    
    # 2. 데이터 처리 루프
    for line in tqdm(lines, desc="Augmenting"):
        entry = json.loads(line)
        
        # 원본 데이터는 그대로 리스트에 추가 (데이터셋 2배 확장을 위해)
        new_entries.append(entry)
        
        # --- 증강 데이터 생성 ---
        aug_entry = entry.copy()
        
        # 이미지 경로 식별 (데이터셋 구조에 따라 키값 확인 필요, 보통 'image' or 'image_path')
        img_filename = entry.get('image') or entry.get('image_path')
        
        if not img_filename:
            continue # 이미지가 없으면 스킵

        # 이미지 파일 찾기 (COCO vs Scannet 경로 처리)
        # 이미지 파일명이 전체 경로인지, 파일명만 있는지에 따라 다름.
        # 일단 현재 폴더 구조상 아래 경로들을 순차적으로 탐색
        potential_paths = [
            os.path.join(BASE_DIR, img_filename),
            os.path.join(BASE_DIR, "val2017", img_filename),
            os.path.join(BASE_DIR, "scannetv2_val", img_filename)
        ]
        
        src_img_path = None
        for path in potential_paths:
            if os.path.exists(path):
                src_img_path = path
                break
        
        if src_img_path is None:
            # 파일을 못 찾으면 증강 포기하고 원본만 유지
            continue

        # 1. 이미지 로드 및 좌우 반전
        try:
            with Image.open(src_img_path) as img:
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 새 파일명 생성 (예: abc.jpg -> abc_flip.jpg)
                name, ext = os.path.splitext(os.path.basename(img_filename))
                new_filename = f"{name}_flip{ext}"
                save_path = os.path.join(AUG_IMG_DIR, new_filename)
                
                # 이미지 저장
                flipped_img.save(save_path)
                
                # 증강된 엔트리에 새 이미지 경로(파일명) 업데이트
                # 학습 로더가 'augmented_images' 폴더도 볼 수 있게 경로 조정 필요
                # 여기서는 상대 경로로 'augmented_images/filename' 저장
                aug_entry['image'] = os.path.join("augmented_images", new_filename)
                
        except Exception as e:
            print(f"Error processing image {src_img_path}: {e}")
            continue

        # 2. 정답(Label/Answer) 반전
        # 데이터셋 포맷에 따라 'answer', 'label', 'conversations' 등 키가 다를 수 있음
        # 일반적인 VQA 포맷인 'answer'라고 가정하고 처리
        if 'answer' in aug_entry:
            aug_entry['answer'] = swap_text(aug_entry['answer'])
            
        # 만약 conversations(LLaVA 포맷) 구조라면 아래 주석 해제하여 사용
        # if 'conversations' in aug_entry:
        #     for conv in aug_entry['conversations']:
        #         if conv['from'] == 'gpt': # 모델의 답변 부분만 수정
        #             conv['value'] = swap_text(conv['value'])

        # 3. 질문(Question) 반전 여부
        # 질문에 "What is on the left?" 같은 표현이 있다면 이것도 바꿔야 함 ("on the right?"으로)
        # 논리적 정합성을 위해 질문도 swap_text 처리 추천
        if 'question' in aug_entry:
            aug_entry['question'] = swap_text(aug_entry['question'])

        # 증강된 엔트리 추가
        new_entries.append(aug_entry)

    # 3. 새로운 JSONL 저장
    print(f"💾 Saving to: {OUTPUT_JSONL}")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"✨ Done! Final dataset size: {len(new_entries)} (Original x 2)")
    print(f"⚠️ Checkpoint: Make sure your dataloader can read images from '{AUG_IMG_DIR}'")

if __name__ == "__main__":
    process_augmentation()