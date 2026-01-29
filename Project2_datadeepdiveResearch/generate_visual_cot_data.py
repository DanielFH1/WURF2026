import json
import os
import re
from tqdm import tqdm

# ================= 설정 =================
BASE_DIR = os.getcwd()
# ★ 중요: 실험 1에서 만든 "빨간 박스 이미지" 데이터셋을 로드
INPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_visual_prompt.jsonl")
OUTPUT_JSONL = os.path.join(BASE_DIR, "data_train_scene_split/train_visual_cot.jsonl")
# =======================================

def extract_objects(text):
    text_lower = text.lower()
    # 기본값
    ref_obj, target_obj = "the reference object", "the target object"

    # 정규식으로 주어/목적어 추출 (Visual Prompt 문구 포함될 수 있음)
    # 예: "Where is the chair in the red bounding box in comparison to..."
    match = re.search(r"location of (?:the |a )?(.+?) in comparison to (?:the |a )?(.+?)\?", text_lower)
    if match: return match.group(2).strip(), match.group(1).strip()
    
    match = re.search(r"perspective of (?:the |a )?(.+?)(?:,| in).+where is (?:the |a )?(.+?)(?:located|positioned|\?)", text_lower)
    if match: return match.group(1).strip(), match.group(2).strip()

    match = re.search(r"which way is (?:the |a )?(.+?) facing", text_lower)
    if match: return "the camera viewpoint", match.group(1).strip()

    return ref_obj, target_obj

def generate_visual_cot_response(question, correct_option):
    """
    Visual CoT 템플릿: 시각적 힌트(Red Box)를 언급하며 추론 유도
    """
    if "." in correct_option:
        answer_label = correct_option.split(".")[0].strip() # "A"
        answer_desc = correct_option.split(".")[1].strip()  # "left"
    else:
        answer_label = correct_option
        answer_desc = "that direction"

    ref, target = extract_objects(question)

    # ★ 핵심 변경점: 템플릿에 "Red Bounding Box" 관련 내용 추가
    cot_template = (
        f"Let's analyze the image step by step with the visual aids. "
        f"1. First, I focus on the area marked with the red bounding box to identify the reference: {ref}. "
        f"2. From this anchored viewpoint, I locate the target object: {target}. "
        f"3. Observing the spatial relationship relative to the red box, {target} is to the {answer_desc}. "
        f"Therefore, the correct option is {answer_label}."
    )
    
    return cot_template

def process():
    print(f"🚀 Generating Visual CoT Dataset (Combining Red Box + Reasoning)...")
    
    # 파일 존재 확인
    if not os.path.exists(INPUT_JSONL):
        print(f"❌ Error: {INPUT_JSONL} 파일이 없습니다. 실험 1(Visual Prompt) 데이터 생성을 먼저 했는지 확인하세요.")
        return

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_entries = []
    
    for line in tqdm(lines):
        entry = json.loads(line)
        new_entry = entry.copy()
        
        question_text = ""
        
        if 'messages' in entry:
            # 질문 찾기
            for msg in entry['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content['type'] == 'text':
                            question_text = content['text']
            
            # 답변 교체 (CoT 적용)
            for msg in new_entry['messages']:
                if msg['role'] == 'assistant':
                    for content in msg['content']:
                        if content['type'] == 'text':
                            original_answer = content['text']
                            # Visual CoT 생성
                            visual_cot = generate_visual_cot_response(question_text, original_answer)
                            content['text'] = visual_cot
                            
        # Legacy 포맷 등은 생략 (train_visual_prompt.jsonl은 messages 포맷임이 확실하므로)

        new_entries.append(new_entry)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"✨ Visual CoT Data Generation Complete!")
    print(f"💾 Saved to: {OUTPUT_JSONL}")
    print("\n[Sample Visual CoT]")
    print(json.dumps(new_entries[0]['messages'][1], indent=2))

if __name__ == "__main__":
    process()