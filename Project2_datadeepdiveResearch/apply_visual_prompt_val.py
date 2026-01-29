import json
import os
import cv2
import re
import torch
from tqdm import tqdm
from ultralytics import YOLOWorld

# ================= CONFIGURATION =================
BASE_DIR = os.getcwd()
# [변경] 입력 파일을 val.jsonl로 변경
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/val.jsonl")
# [변경] 출력 파일명
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/val_visual_prompt.jsonl")

print("🚀 Loading YOLO-World Model (CPU Mode)...")
model = YOLOWorld('yolov8s-worldv2.pt')
# =================================================

def extract_target_info(text):
    text_lower = text.lower()
    match = re.search(r"(?:imagine you're|as|picture yourself as) (?:the |a |this )?(.+?) (?:in|looking|facing|within|photo)", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    match = re.search(r"from (?:the )?perspective of (?:the |this |a )?(.+?)(?:, | \?|\.|$)", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    match = re.search(r"from (?:the |this |a )?(.+?)'s perspective", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    match = re.search(r"(?:respect to|comparison to) (?:the |this |a )?(.+?)(?:\?|\.| in|$)", text_lower)
    if match: return match.group(1).strip(), match.group(1).strip()
    return None, None

def draw_dynamic_box(image_path, target_class, save_path):
    try:
        model.set_classes([target_class])
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

def process_visual_prompt():
    print(f"🚀 Generating Visual Prompts for Validation Set...")
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_entries = []
    stats = {"success": 0, "skip": 0}

    for line in tqdm(lines, desc="Processing Val"):
        entry = json.loads(line)
        new_entry = entry.copy()
        
        img_rel_path = None
        target_text_obj = None
        target_class = None
        replace_span = None
        
        # 구조 파악 (Messages vs Question/Image)
        if 'messages' in entry:
            for msg in new_entry['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content['type'] == 'image': img_rel_path = content['image']
                        if content['type'] == 'text':
                            t_cls, t_span = extract_target_info(content['text'])
                            if t_cls:
                                target_class = t_cls
                                replace_span = t_span
                                target_text_obj = content
        elif 'image' in entry and 'question' in entry:
            img_rel_path = entry['image']
            t_cls, t_span = extract_target_info(entry['question'])
            if t_cls:
                target_class = t_cls
                replace_span = t_span

        if not target_class or not img_rel_path:
            new_entries.append(entry)
            stats["skip"] += 1
            continue

        real_image_path = os.path.join(BASE_DIR, img_rel_path)
        # 검증용 이미지는 별도 폴더에 저장 (충돌 방지)
        save_rel_path = os.path.join("visual_prompt_images_val", img_rel_path)
        save_full_path = os.path.join(BASE_DIR, save_rel_path)

        if draw_dynamic_box(real_image_path, target_class, save_full_path):
            suffix = " in the red bounding box"
            try:
                if 'messages' in new_entry:
                    pattern = re.compile(re.escape(replace_span), re.IGNORECASE)
                    new_entry['messages'][0]['content'][1]['text'] = pattern.sub(f"{replace_span}{suffix}", target_text_obj['text'], count=1)
                    new_entry['messages'][0]['content'][0]['image'] = save_rel_path
                elif 'question' in new_entry:
                    pattern = re.compile(re.escape(replace_span), re.IGNORECASE)
                    new_entry['question'] = pattern.sub(f"{replace_span}{suffix}", new_entry['question'], count=1)
                    new_entry['image'] = save_rel_path
                
                new_entries.append(new_entry)
                stats["success"] += 1
            except:
                new_entries.append(entry)
        else:
            new_entries.append(entry)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"✨ Validation Set 변환 완료: {stats['success']}개 적용됨")
    print(f"💾 저장 경로: {OUTPUT_JSONL}")

if __name__ == "__main__":
    process_visual_prompt()