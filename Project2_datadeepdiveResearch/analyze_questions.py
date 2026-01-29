import json
import os
from collections import Counter

# ================= CONFIGURATION =================
BASE_DIR = os.getcwd()
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train.jsonl")
OUTPUT_TXT = os.path.join(BASE_DIR, "all_questions.txt")
# =================================================

def analyze():
    print(f"🚀 Analyzing questions from: {INPUT_JSONL}")
    
    questions = []
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            
            # 질문 텍스트 추출
            text = ""
            if 'messages' in entry:
                for msg in entry['messages']:
                    if msg['role'] == 'user':
                        for content in msg['content']:
                            if content['type'] == 'text':
                                text = content['text']
                                break
            elif 'question' in entry:
                text = entry['question']
            
            if text:
                questions.append(text)

    # 1. 파일로 저장
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(q + "\n")
            
    print(f"💾 Saved all questions to: {OUTPUT_TXT}")
    
    # 2. 통계 분석
    total = len(questions)
    print("\n" + "="*40)
    print(f"📊 Question Statistics (Total: {total})")
    print("="*40)
    
    # 주요 키워드 포함 여부 카운팅
    keywords = {
        "person": 0,
        "viewpoint": 0,
        "perspective": 0,
        "facing": 0,
        "camera": 0,
        "comparison": 0, # 사물 간 비교
        "located": 0,
        "where": 0
    }
    
    for q in questions:
        q_lower = q.lower()
        for k in keywords:
            if k in q_lower:
                keywords[k] += 1
                
    # 통계 출력
    for k, v in keywords.items():
        print(f" - '{k}': {v} ({v/total*100:.1f}%)")
        
    print("-" * 40)
    print("🔍 Sample Questions (First 20):")
    for i, q in enumerate(questions[:20]):
        print(f"{i+1}. {q.replace(chr(10), ' ')}") # 줄바꿈 제거해서 출력

if __name__ == "__main__":
    analyze()