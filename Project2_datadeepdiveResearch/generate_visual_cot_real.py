import json
import os
import base64
import requests
import concurrent.futures
from tqdm import tqdm

# ================= 설정 =================
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct" 
API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "EMPTY" 

BASE_DIR = os.getcwd()
# ★ 입력: 실험 1에서 만든 "빨간 박스 이미지" 데이터셋
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_visual_prompt.jsonl")
# ★ 출력: Visual CoT 데이터셋
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_visual_cot_real.jsonl")

# 이미지 경로 prefix (train_visual_prompt.jsonl은 상대경로일 수 있음)
# 보통 generate_visual_prompt.py로 만들면 현재 폴더 기준 상대경로로 저장됨
# 필요시 절대경로로 수정
# ====================================================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_reasoning(entry):
    try:
        # 1. 데이터 파싱
        image_rel_path = ""
        question = ""
        correct_answer = ""
        
        if 'messages' in entry:
            for msg in entry['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content['type'] == 'image': image_rel_path = content['image']
                        if content['type'] == 'text': question = content['text']
                if msg['role'] == 'assistant':
                    for content in msg['content']:
                        if content['type'] == 'text': correct_answer = content['text']
        
        # 이미지 경로 처리 (Visual Prompt 이미지는 보통 'visual_prompt_images/...' 에 있음)
        full_img_path = os.path.join(BASE_DIR, image_rel_path)
        
        if not os.path.exists(full_img_path):
            # 혹시 경로가 안 맞으면 체크
            print(f"Skipping missing image: {full_img_path}")
            return None

        base64_image = encode_image(full_img_path)

        # ★ 핵심: Visual CoT를 위한 시스템 프롬프트
        system_prompt = (
            "You are an expert in spatial reasoning. "
            "The image provided contains a **RED BOUNDING BOX** drawn around a reference object. "
            "Use this visual cue to anchor your reasoning."
        )
        
        user_text = (
            f"Question: {question}\n"
            f"Correct Answer: {correct_answer}\n\n"
            f"Please generate a 'Visual Chain-of-Thought' explanation.\n"
            f"1. First, explicitly mention the object inside the **red bounding box**.\n"
            f"2. Describe the spatial relationship of the target object relative to this red box.\n"
            f"3. Conclude logically why the answer is {correct_answer}."
        )

        # 3. Request 보내기
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload).json()
        
        if 'choices' not in response:
            return None
            
        reasoning = response['choices'][0]['message']['content']
        
        # 4. 결과 저장
        new_entry = entry.copy()
        new_entry['messages'][1]['content'][0]['text'] = reasoning
        return new_entry

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"🚀 Generating Visual CoT Data (Red Box + Reasoning)...")
    
    if not os.path.exists(INPUT_JSONL):
        print(f"❌ Error: {INPUT_JSONL} 파일이 없습니다. 실험 1(Visual Prompt) 데이터 생성을 먼저 했는지 확인하세요.")
        return

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    results = []
    
    # 병렬 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_reasoning, json.loads(line)) for line in tqdm(lines, desc="Scheduling")]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines), desc="Generating"):
            res = future.result()
            if res:
                results.append(res)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
            
    print(f"✨ Visual CoT Data Generation Complete! Saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()