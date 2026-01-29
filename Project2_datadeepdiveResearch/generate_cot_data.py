import json
import os
import re
from tqdm import tqdm

# ================= 설정 =================
BASE_DIR = os.getcwd()
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_cot.jsonl")
# =======================================

def extract_objects(text):
    """
    질문에서 Reference(기준)와 Target(대상)을 추출 (이전 코드 재활용 및 개선)
    """
    text_lower = text.lower()
    ref_obj, target_obj = "the reference object", "the target object"

    # 패턴 1: Comparison (Where is X in comparison to Y?) -> Ref: Y, Target: X
    match = re.search(r"location of (?:the |a )?(.+?) in comparison to (?:the |a )?(.+?)\?", text_lower)
    if match: return match.group(2).strip(), match.group(1).strip()
    
    # 패턴 2: Perspective (From the perspective of X, where is Y?) -> Ref: X, Target: Y
    match = re.search(r"perspective of (?:the |a )?(.+?)(?:,| in).+where is (?:the |a )?(.+?)(?:located|positioned|\?)", text_lower)
    if match: return match.group(1).strip(), match.group(2).strip()

    # 패턴 3: Facing (Which way is X facing?) -> Ref: Camera/Self, Target: X
    match = re.search(r"which way is (?:the |a )?(.+?) facing", text_lower)
    if match: return "the camera viewpoint", match.group(1).strip()

    return ref_obj, target_obj # 추출 실패 시 기본값

def generate_cot_response(question, correct_option, options_text):
    """
    정답(A/B/C/D)을 바탕으로 논리적인 해설 생성
    """
    # 1. 정답 텍스트 파싱 (예: "A. left" -> "left")
    answer_text = "unknown direction"
    
    # 옵션 텍스트에서 정답 내용 추출 (예: "A. left\nB. right...")
    # options_text는 질문 뒤에 붙어있으므로 분리 필요하지만, 
    # 여기서는 간단히 correct_option ("A. left") 자체를 이용
    if "." in correct_option:
        answer_label = correct_option.split(".")[0].strip() # "A"
        answer_desc = correct_option.split(".")[1].strip()  # "left"
    else:
        answer_label = correct_option # "A"
        answer_desc = "that direction"

    # 2. 객체 추출
    ref, target = extract_objects(question)

    # 3. CoT 템플릿 작성 (Step-by-Step Thinking)
    # 모델에게 '좌표' 개념을 심어주기 위한 가상의 표현 사용
    cot_template = (
        f"Let's think step by step to determine the spatial relationship. "
        f"1. First, I identify the reference point: {ref}. "
        f"2. Next, I locate the target object: {target}. "
        f"3. By analyzing their relative positions in the 3D space, {target} is positioned to the {answer_desc} of {ref}. "
        f"Therefore, the correct option is {answer_label}."
    )
    
    return cot_template

def process_cot():
    print(f"🚀 Generating Chain-of-Thought Dataset...")
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_entries = []
    
    for line in tqdm(lines):
        entry = json.loads(line)
        new_entry = entry.copy()
        
        # 데이터 구조 파악
        question_text = ""
        answer_text = ""
        
        if 'messages' in entry:
            # 질문 찾기
            for msg in entry['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content['type'] == 'text':
                            question_text = content['text']
            # 정답 찾기 & 교체
            for msg in new_entry['messages']:
                if msg['role'] == 'assistant':
                    for content in msg['content']:
                        if content['type'] == 'text':
                            original_answer = content['text'] # 예: "B. front"
                            # CoT 생성
                            cot_answer = generate_cot_response(question_text, original_answer, question_text)
                            content['text'] = cot_answer
                            
        elif 'question' in entry and 'answer' in entry: # Legacy format
            question_text = entry['question']
            original_answer = entry['answer'] # Legacy는 보통 정답 라벨만 있거나 함. 확인 필요.
            # Legacy 포맷은 복잡하므로 messages 포맷 위주로 처리 가정
            # 만약 Legacy 데이터가 섞여있다면 여기서 처리
            cot_answer = generate_cot_response(question_text, original_answer, question_text)
            new_entry['answer'] = cot_answer

        new_entries.append(new_entry)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"✨ CoT Data Generation Complete!")
    print(f"💾 Saved to: {OUTPUT_JSONL}")
    
    # 샘플 출력
    print("\n[Sample CoT Data]")
    print(json.dumps(new_entries[0]['messages'][1], indent=2))

if __name__ == "__main__":
    process_cot()