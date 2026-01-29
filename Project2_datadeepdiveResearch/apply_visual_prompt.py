import json
import os
import cv2
import re
import torch
from tqdm import tqdm
from ultralytics import YOLOWorld

# ================= CONFIGURATION =================
BASE_DIR = os.getcwd()
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_visual_prompt.jsonl")

# [핵심] GPU 에러 방지를 위해 CPU 강제 사용
print("🚀 Loading YOLO-World Model (CPU Mode)...")
model = YOLOWorld('yolov8s-worldv2.pt')

# =================================================

def extract_target_info(text):
    """
    질문 텍스트를 분석하여 타겟 클래스와 교체할 텍스트 구간을 추출합니다.
    """
    text_lower = text.lower()
    
    # 1. 역할 이입형 ("Imagine you're the X", "As the X")
    match = re.search(r"(?:imagine you're|as|picture yourself as) (?:the |a |this )?(.+?) (?:in|looking|facing|within|photo)", text_lower)
    if match:
        raw_obj = match.group(1).strip()
        return raw_obj, raw_obj

    # 2. 관점 명시형 ("From the perspective of the X")
    match = re.search(r"from (?:the )?perspective of (?:the |this |a )?(.+?)(?:, | \?|\.|$)", text_lower)
    if match:
        raw_obj = match.group(1).strip()
        return raw_obj, raw_obj
    
    # 2-1. 소유격 관점 ("From this woman's perspective")
    match = re.search(r"from (?:the |this |a )?(.+?)'s perspective", text_lower)
    if match:
        raw_obj = match.group(1).strip()
        return raw_obj, raw_obj

    # 3. 사물 간 비교형 ("comparison to the X")
    match = re.search(r"(?:respect to|comparison to) (?:the |this |a )?(.+?)(?:\?|\.| in|$)", text_lower)
    if match:
        raw_obj = match.group(1).strip()
        return raw_obj, raw_obj

    return None, None

def draw_dynamic_box(image_path, target_class, save_path):
    """
    YOLO에게 target_class를 찾게 하고 빨간 박스를 그립니다. (CPU 모드)
    """
    try:
        # [핵심] 클래스 설정
        model.set_classes([target_class])
        
        # [핵심] device='cpu'를 명시해서 CUDA 에러 원천 차단
        results = model.predict(image_path, conf=0.05, verbose=False, device='cpu')
        result = results[0]

        if len(result.boxes) == 0:
            return False

        # 가장 신뢰도 높은 객체 선택
        best_box = sorted(result.boxes, key=lambda x: x.conf[0], reverse=True)[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        img = cv2.imread(image_path)
        if img is None: 
            print(f"[OpenCV Fail] Cannot read image: {image_path}")
            return False
        
        # 빨간 박스 그리기 (BGR)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        return True

    except Exception as e:
        # 에러가 나면 뭔지 출력
        print(f"\n[Error processing {target_class}] {e}")
        return False

def process_visual_prompt():
    print(f"🚀 Starting Dynamic Visual Prompt Generation (Robust Mode)...")
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_entries = []
    stats = {
        "total": len(lines),
        "success": 0,       
        "skip_no_target": 0, 
        "fail_detection": 0 
    }

    for line in tqdm(lines, desc="Processing"):
        entry = json.loads(line)
        new_entry = entry.copy()
        
        img_rel_path = None
        target_text_obj = None
        target_class = None
        replace_span = None
        
        # 1. 텍스트 분석 및 타겟 추출
        if 'messages' in entry:
            for msg in new_entry['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content['type'] == 'image':
                            img_rel_path = content['image']
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
        
        # 2. 타겟 없음 스킵
        if not target_class or not img_rel_path:
            new_entries.append(entry)
            stats["skip_no_target"] += 1
            continue

        # 3. 이미지 처리
        real_image_path = os.path.join(BASE_DIR, img_rel_path)
        save_rel_path = os.path.join("visual_prompt_images", img_rel_path)
        save_full_path = os.path.join(BASE_DIR, save_rel_path)

        is_detected = draw_dynamic_box(real_image_path, target_class, save_full_path)

        if is_detected:
            # 4. 텍스트 수정
            suffix = " in the red bounding box"
            try:
                if 'messages' in new_entry:
                    original_text = target_text_obj['text']
                    pattern = re.compile(re.escape(replace_span), re.IGNORECASE)
                    new_text = pattern.sub(f"{replace_span}{suffix}", original_text, count=1)
                    target_text_obj['text'] = new_text
                    
                    # 이미지 경로 업데이트
                    for msg in new_entry['messages']:
                        if msg['role'] == 'user':
                            for content in msg['content']:
                                if content['type'] == 'image':
                                    content['image'] = save_rel_path
                                    
                elif 'question' in new_entry:
                    original_text = new_entry['question']
                    pattern = re.compile(re.escape(replace_span), re.IGNORECASE)
                    new_entry['question'] = pattern.sub(f"{replace_span}{suffix}", original_text, count=1)
                    new_entry['image'] = save_rel_path
                
                new_entries.append(new_entry)
                stats["success"] += 1
            except Exception as e:
                print(f"[Text Mod Error] {e}")
                new_entries.append(entry)
                stats["fail_detection"] += 1
        else:
            new_entries.append(entry)
            stats["fail_detection"] += 1

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n✨ 완료!")
    print(f" - 총 데이터: {stats['total']}")
    print(f" - [성공] Visual Prompt 적용: {stats['success']} (이 숫자가 중요함)")
    print(f" - [제외] 타겟 없음: {stats['skip_no_target']}")
    print(f" - [실패] YOLO 감지 실패: {stats['fail_detection']}")
    print(f"💾 저장 경로: {OUTPUT_JSONL}")

if __name__ == "__main__":
    process_visual_prompt()