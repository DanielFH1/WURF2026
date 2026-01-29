import json
import os
import random
import re
from collections import defaultdict

# ================= 설정 =================
INPUT_FILE = "eval/ViewSpatial-Bench.json"  # 원본 데이터 경로
OUTPUT_DIR = "data_train_scene_split"       # 저장될 폴더
SEED = 42

# 비율 설정 (8:1:1)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# ========================================

def format_to_qwen_chat(item):
    """
    Raw 데이터를 Qwen 학습용 'messages' 포맷으로 변환
    """
    # 1. 이미지 경로 처리
    image_path = item['image_path']
    if isinstance(image_path, list):
        image_path = image_path[0]  # 리스트면 첫 번째 꺼내기
    
    # 2. 질문 텍스트 구성
    # 질문 + 보기 + 지시사항
    question_text = f"{item['question']}\n{item['choices']}\nAnswer with the option letter."
    
    # 3. Messages 구조 생성
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question_text}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": item['answer']}
            ]
        }
    ]
    
    return {"messages": messages}

def get_scene_id(item):
    img_path = item['image_path'][0] if isinstance(item['image_path'], list) else item['image_path']
    match = re.search(r'(scene\d+_\d+)', img_path)
    if match:
        return match.group(1)
    return os.path.basename(img_path)

def split_and_convert():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📂 Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # 1. Scene별 그룹화
    scene_dict = defaultdict(list)
    for item in data:
        scene_id = get_scene_id(item)
        scene_dict[scene_id].append(item)
    
    unique_scenes = list(scene_dict.keys())
    random.shuffle(unique_scenes)
    
    # 2. Split 계산
    n_scenes = len(unique_scenes)
    n_train = int(n_scenes * TRAIN_RATIO)
    n_val = int(n_scenes * VAL_RATIO)
    
    train_scenes = unique_scenes[:n_train]
    val_scenes = unique_scenes[n_train:n_train+n_val]
    test_scenes = unique_scenes[n_train+n_val:]
    
    # 3. 데이터 변환 및 저장 함수
    def save_converted(scenes, filename, is_jsonl=True):
        converted_data = []
        for sc in scenes:
            for item in scene_dict[sc]:
                # ★ 여기서 변환 수행!
                converted_item = format_to_qwen_chat(item)
                converted_data.append(converted_item)
        
        path = os.path.join(OUTPUT_DIR, filename)
        
        if is_jsonl:
            with open(path, 'w') as f:
                for entry in converted_data:
                    f.write(json.dumps(entry) + '\n')
        else:
            # Test용은 평가 코드 호환성을 위해 원본 포맷 유지 (변환 X)
            # 평가 코드는 보통 원본 구조를 기대하므로, Raw 데이터를 그대로 저장
            raw_data = []
            for sc in scenes:
                raw_data.extend(scene_dict[sc])
            with open(path, 'w') as f:
                json.dump(raw_data, f, indent=4)
                
        print(f"✅ Saved {filename}: {len(converted_data) if is_jsonl else len(raw_data)} items")

    print("\n--- Converting & Splitting ---")
    save_converted(train_scenes, "train.jsonl", is_jsonl=True)  # 학습용: 변환 O
    save_converted(val_scenes, "val.jsonl", is_jsonl=True)      # 검증용: 변환 O
    save_converted(test_scenes, "test.json", is_jsonl=False)    # 평가용: 변환 X (원본 유지)

    print("\n🚀 Data preparation complete!")

if __name__ == "__main__":
    split_and_convert()