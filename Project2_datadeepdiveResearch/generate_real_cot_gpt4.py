import json
import os
import base64
import time
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

# ================= 설정 =================
# 1. 사용할 Teacher 모델 (API 또는 로컬 72B 모델)
# API 사용 시: "gpt-4o"
# 로컬 사용 시 (vLLM): "Qwen/Qwen2-VL-72B-Instruct" (모델명은 서버 설정 따름)
MODEL_NAME = "gpt-4o" 
API_KEY = "sk-..."  # 실제 키 입력 필요
BASE_URL = "https://api.openai.com/v1" # 로컬 vLLM 사용 시: "http://localhost:8000/v1"

# 2. 데이터 경로
BASE_DIR = os.getcwd()
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train.jsonl") # 원본 (이미지+질문+단답)
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_real_cot.jsonl") # 결과물

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
# =======================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_reasoning(entry):
    """
    Teacher Model에게 정답을 주고 추론 과정을 생성하게 함
    """
    try:
        # 데이터 파싱
        image_rel_path = ""
        question = ""
        correct_answer = ""
        
        # messages 포맷 파싱
        if 'messages' in entry:
            for msg in entry['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content['type'] == 'image': image_rel_path = content['image']
                        if content['type'] == 'text': question = content['text']
                if msg['role'] == 'assistant':
                    for content in msg['content']:
                        if content['type'] == 'text': correct_answer = content['text']
        
        full_img_path = os.path.join(BASE_DIR, image_rel_path)
        if not os.path.exists(full_img_path): return None

        base64_image = encode_image(full_img_path)

        # ★ Teacher를 위한 프롬프트 (Reverse Reasoning)
        system_prompt = (
            "You are an expert in spatial reasoning and 3D perception. "
            "I will provide an image, a question, and the CORRECT answer. "
            "Your task is to generate a 'Chain-of-Thought' explanation that logically leads to that answer.\n"
            "Rules:\n"
            "1. Start by identifying the reference object and the target object in the image.\n"
            "2. Describe their positions relative to each other or the camera.\n"
            "3. Conclude with 'Therefore, the correct option is X.'\n"
            "4. Keep the explanation concise but strictly logical based on visual evidence."
        )
        
        user_content = [
            {"type": "text", "text": f"Question: {question}\nCorrect Answer: {correct_answer}\n\nExplain why this is the correct answer step-by-step."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=300
        )
        
        reasoning = response.choices[0].message.content
        
        # 결과 저장용 새로운 엔트리 생성
        new_entry = entry.copy()
        # Assistant의 답변을 Teacher가 생성한 Reasoning으로 교체
        new_entry['messages'][1]['content'][0]['text'] = reasoning
        
        return new_entry

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"🚀 Generating Real CoT Data using {MODEL_NAME}...")
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 테스트용으로 10개만 먼저 해보고 싶으면: lines = lines[:10]
    
    results = []
    # 속도를 위해 병렬 처리 (API 사용 시) / 로컬 GPU면 max_workers=1 추천
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_reasoning, json.loads(line)) for line in lines]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines)):
            res = future.result()
            if res:
                results.append(res)

    # 저장
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
            
    print(f"✨ 완료! 저장 경로: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()