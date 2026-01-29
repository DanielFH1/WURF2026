import json
import os
import cv2
import re
from tqdm import tqdm
from ultralytics import YOLOWorld

# ================= 설정 =================
BASE_DIR = os.getcwd()
INPUT_JSON = os.path.join(BASE_DIR, "data_train_scene_split/test.json") 
OUTPUT_JSON = os.path.join(BASE_DIR, "data_train_scene_split/test_visual_prompt.json")

print("🚀 Loading YOLO-World (CPU Mode)...")
model = YOLOWorld('yolov8s-worldv2.pt')
# =======================================

def extract_target_info(text):
    text_lower = text.lower()
    # 1. 역할 이입형
    match = re.search(r"(?:imagine you're|as|picture yourself as) (?:the |a |this )?(.+?) (?:in|looking|facing|within|photo)", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    # 2. 관점 명시형
    match = re.search(r"from (?:the )?perspective of (?:the |this |a )?(.+?)(?:, | \?|\.|$)", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    match = re.search(r"from (?:the |this |a )?(.+?)'s perspective", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    # 3. 비교형
    match = re.search(r"(?:respect to|comparison to) (?:the |this |a )?(.+?)(?:\?|\.| in|$)", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    return None, None

def draw_dynamic_box(image_path, target_class, save_path):
    try:
        model.set_classes([target_class])
        # CPU 모드
        results = model.predict(image_path, conf=0.05, verbose=False, device='cpu')
        result = results[0]
        if len(result.boxes) == 0: return False
        
        best_box = sorted(result.boxes, key=lambda x: x.conf[0], reverse=True)[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        
        img = cv2.imread(image_path)
        if img is None: return False
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        return True
    except: return False

def process():
    print(f"🚀 Processing Test Set for Evaluation (Fix List Type)...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    stats = {"success": 0, "skip": 0}

    for item in tqdm(data):
        new_item = item.copy()
        
        # [수정] image_path가 리스트일 경우 첫 번째 요소 추출
        img_rel_path_raw = item.get('image_path')
        if isinstance(img_rel_path_raw, list):
            img_rel_path = img_rel_path_raw[0]
        else:
            img_rel_path = img_rel_path_raw
            
        question = item.get('question')
        
        target_class, replace_span = extract_target_info(question)
        
        if not target_class or not img_rel_path:
            new_data.append(item)
            stats["skip"] += 1
            continue
            
        real_image_path = os.path.join(BASE_DIR, img_rel_path)
        save_rel_path = os.path.join("visual_prompt_images_test", img_rel_path)
        save_full_path = os.path.join(BASE_DIR, save_rel_path)
        
        if draw_dynamic_box(real_image_path, target_class, save_full_path):
            # 성공 시 이미지 경로 교체 (원본 포맷 유지: 리스트면 리스트로)
            if isinstance(img_rel_path_raw, list):
                new_item['image_path'] = [save_rel_path]
            else:
                new_item['image_path'] = save_rel_path
            
            suffix = " in the red bounding box"
            pattern = re.compile(re.escape(replace_span), re.IGNORECASE)
            new_item['question'] = pattern.sub(f"{replace_span}{suffix}", question, count=1)
            
            new_data.append(new_item)
            stats["success"] += 1
        else:
            new_data.append(item)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)
        
    print(f"✨ Test Set 변환 완료: {stats['success']}개 적용됨")
    print(f"💾 저장 경로: {OUTPUT_JSON}")

if __name__ == "__main__":
    process()