import json
import os
from PIL import Image
from tqdm import tqdm

# ================= CONFIGURATION =================
# 현재 작업 경로 (/nas_data2/seungwoo/2/ViewSpatial-Bench)
BASE_DIR = os.getcwd()

# 파일 경로 설정
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_augmented.jsonl")
AUG_IMG_DIR = os.path.join(BASE_DIR, "augmented_images")

# 텍스트 반전 매핑
FLIP_MAPPING = {
    "left": "right", "right": "left",
    "front-left": "front-right", "front-right": "front-left",
    "back-left": "back-right", "back-right": "back-left",
    "Left": "Right", "Right": "Left",
    "Front-Left": "Front-Right", "Front-Right": "Front-Left",
    "Back-Left": "Back-Right", "Back-Right": "Back-Left",
    "LEFT": "RIGHT", "RIGHT": "LEFT",
    # A. left, B. right 같은 보기를 위한 매핑
    "A. left": "A. right", "B. left": "B. right", "C. left": "C. right", "D. left": "D. right",
    "A. right": "A. left", "B. right": "B. left", "C. right": "C. left", "D. right": "D. left"
}
# =================================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def swap_text(text):
    """단순 단어 치환 방식 (구두점 처리 포함)"""
    words = text.split()
    new_words = []
    for word in words:
        # 구두점 분리 (예: "left." -> "left", ".")
        clean_word = word.strip(".,?!:;")
        prefix = word[:word.find(clean_word)] if clean_word else ""
        suffix = word[len(prefix)+len(clean_word):]

        if clean_word in FLIP_MAPPING:
            new_word = prefix + FLIP_MAPPING[clean_word] + suffix
        else:
            new_word = word
        new_words.append(new_word)
    return " ".join(new_words)

def process_augmentation():
    print(f"🚀 Starting Horizontal Flip Augmentation (Fixed Path)...")
    print(f"📂 Current DIR: {BASE_DIR}")
    
    ensure_dir(AUG_IMG_DIR)
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"📊 Total samples: {len(lines)}")
    
    new_entries = []
    success_count = 0
    skip_count = 0
    
    for line in tqdm(lines, desc="Augmenting"):
        entry = json.loads(line)
        new_entries.append(entry) # 원본 유지
        
        aug_entry = entry.copy()
        
        # 1. 이미지 경로 추출
        # JSON 예시: "ViewSpatial-Bench/val2017/000000380711.jpg"
        img_rel_path = None
        
        # messages 구조 확인 (User/Assistant ChatML format)
        if 'messages' in entry:
            for msg in entry['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content['type'] == 'image':
                            img_rel_path = content['image']
                            break
        # 일반적인 key 확인
        if not img_rel_path:
            img_rel_path = entry.get('image')

        if not img_rel_path:
            # 이미지가 없는 데이터면 스킵
            skip_count += 1
            continue

        # 2. 실제 이미지 경로 찾기 (단순 결합)
        # /nas.../ViewSpatial-Bench + / + ViewSpatial-Bench/val2017/...
        real_image_path = os.path.join(BASE_DIR, img_rel_path)
        
        if not os.path.exists(real_image_path):
            # 혹시 경로가 틀릴 경우를 대비한 Fallback (파일명만으로 찾기 - 위험하지만 시도)
            # 하지만 Scannet 때문에 경로 유지가 중요하므로 로그만 찍고 스킵
            # print(f"[Missing] {real_image_path}") 
            skip_count += 1
            continue

        # 3. 이미지 Flip 및 저장
        try:
            with Image.open(real_image_path) as img:
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 저장 경로 생성 (augmented_images/ViewSpatial-Bench/val2017/...)
                # 원본 폴더 구조를 그대로 유지해야 안전함
                save_rel_path = img_rel_path 
                
                # 파일명 변경 (abc.jpg -> abc_flip.jpg)
                base, ext = os.path.splitext(save_rel_path)
                save_rel_path_flip = f"{base}_flip{ext}"
                
                # 최종 저장 절대 경로
                save_full_path = os.path.join(AUG_IMG_DIR, save_rel_path_flip)
                
                # 폴더 생성
                os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
                
                # 저장
                flipped_img.save(save_full_path)
                
                # JSON 업데이트 (경로는 augmented_images 부터 시작하도록)
                # 예: augmented_images/ViewSpatial-Bench/val2017/000000380711_flip.jpg
                new_json_path = os.path.join("augmented_images", save_rel_path_flip)
                
                # messages 구조 업데이트
                if 'messages' in aug_entry:
                    for msg in aug_entry['messages']:
                        if msg['role'] == 'user':
                            for content in msg['content']:
                                if content['type'] == 'image':
                                    content['image'] = new_json_path
                else:
                    aug_entry['image'] = new_json_path

        except Exception as e:
            print(f"Error processing {real_image_path}: {e}")
            skip_count += 1
            continue

        # 4. 텍스트 반전 (Question & Answer)
        if 'messages' in aug_entry:
            for msg in aug_entry['messages']:
                if isinstance(msg['content'], list):
                    for content in msg['content']:
                        if content['type'] == 'text':
                            content['text'] = swap_text(content['text'])
                elif isinstance(msg['content'], str):
                     msg['content'] = swap_text(msg['content'])
        
        # Legacy 포맷 대응
        if 'answer' in aug_entry:
            aug_entry['answer'] = swap_text(aug_entry['answer'])
        if 'question' in aug_entry:
            aug_entry['question'] = swap_text(aug_entry['question'])

        new_entries.append(aug_entry)
        success_count += 1

    # 저장
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\n✨ 최종 완료!")
    print(f" - 성공: {success_count} / {len(lines)}")
    print(f" - 실패(경로없음 등): {skip_count}")
    print(f" - 저장됨: {OUTPUT_JSONL}")
    print(f"⚠️ [Check] 학습 config에서 image_folder 경로를 '{BASE_DIR}'로 설정하면 됩니다.")

if __name__ == "__main__":
    process_augmentation()