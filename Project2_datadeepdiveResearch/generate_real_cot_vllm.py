import json
import os
import base64
import requests
import concurrent.futures
from tqdm import tqdm

# ================= 설정 =================
# vLLM 서버에서 띄운 모델명과 정확히 일치해야 함
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
# 만약 7B를 띄웠다면: "Qwen/Qwen2-VL-7B-Instruct"

API_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "EMPTY"  # vLLM은 키 불필요

BASE_DIR = os.getcwd()
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_real_cot.jsonl")
# =======================================

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
        
        full_img_path = os.path.join(BASE_DIR, image_rel_path)
        if not os.path.exists(full_img_path): return None

        base64_image = encode_image(full_img_path)

        # 2. Teacher에게 보낼 프롬프트 (Reverse Reasoning)
        system_prompt = (
            "You are an expert in spatial reasoning. "
            "I will give you an image, a question, and the answer. "
            "Your job is to explain the step-by-step reasoning that leads to that answer."
        )
        
        user_text = (
            f"Question: {question}\n"
            f"Correct Answer: {correct_answer}\n\n"
            f"Please generate a concise 'Chain-of-Thought' explanation.\n"
            f"1. Identify the reference and target objects.\n"
            f"2. Analyze their spatial relationship relative to the camera or each other.\n"
            f"3. Conclude logically why the answer is {correct_answer}."
        )

        # 3. Request 보내기 (openai 라이브러리 대신 requests 사용으로 가볍게 처리)
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
            print(f"Error response: {response}")
            return None
            
        reasoning = response['choices'][0]['message']['content']
        
        # 4. 결과 저장
        new_entry = entry.copy()
        new_entry['messages'][1]['content'][0]['text'] = reasoning
        return new_entry

    except Exception as e:
        print(f"Error processing entry: {e}")
        return None

def main():
    print(f"🚀 Generating Real CoT Data using {MODEL_NAME} on GPU 3...")
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    results = []
    
    # vLLM은 배칭을 잘하므로 Thread를 늘려도 됨 (GPU 3번 부하를 최대로)
    # 48GB 메모리 꽉 채워 쓰도록 worker 수를 조절하세요.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(generate_reasoning, json.loads(line)) for line in tqdm(lines, desc="Scheduling")]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines), desc="Generating"):
            res = future.result()
            if res:
                results.append(res)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
            
    print(f"✨ Data Generation Complete! Saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()