import json
import os

# ================= 설정 =================
BASE_DIR = os.getcwd()
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_visual_prompt.jsonl")
# =======================================

def inspect():
    print(f"🔍 Inspecting Scannet Visual Prompts (Showing Full Paths)")
    print("="*80)

    count = 0
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            # 1. 텍스트와 이미지 경로 추출
            text = ""
            img_rel_path = ""
            
            if 'messages' in entry:
                for msg in entry['messages']:
                    if msg['role'] == 'user':
                        for content in msg['content']:
                            if content['type'] == 'text':
                                text = content['text']
                            if content['type'] == 'image':
                                img_rel_path = content['image']
            elif 'question' in entry:
                text = entry['question']
                img_rel_path = entry['image']

            # 2. 필터링 조건 (Scannet + Red Box)
            if "scannet" in img_rel_path.lower() and "red bounding box" in text.lower():
                count += 1
                
                # 절대 경로 생성
                abs_path = os.path.join(BASE_DIR, img_rel_path)
                
                print(f"[{count}]")
                print(f"📂 Relative Path: {img_rel_path}")
                print(f"📍 Absolute Path: {abs_path}")
                print(f"❓ Question: {text}")
                print("-" * 80)
                
            if count >= 20:
                break
    
    if count == 0:
        print("⚠️ 변환된 Scannet 데이터를 찾을 수 없습니다.")

if __name__ == "__main__":
    inspect()