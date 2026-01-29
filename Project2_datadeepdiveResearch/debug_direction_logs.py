import torch
import json
import os
import re
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 설정 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "./checkpoints/mvsm_baseline_merged"
TEST_FILE = "data_train_scene_split/test.json"
BASE_IMAGE_DIR = "/nas_data2/seungwoo/2/ViewSpatial-Bench"
RESULT_DIR = "data_divedive_results"
os.makedirs(RESULT_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULT_DIR, "debug_direction_logs.txt")

# 분석 대상 Task
TARGET_TASKS = [
    "Camera perspective - Relative Direction",
    "Person perspective - Relative Direction",
    "Person perspective - Scene Simulation Relative Direction"
]
# ========================================

def extract_direction_debug(text):
    """디버깅용: 어떤 단어로 매칭되었는지 확인"""
    text_lower = text.lower().replace("-", " ").strip()
    
    compound_map = {
        "front left": "front-left", "front right": "front-right",
        "back left": "back-left", "back right": "back-right",
        "left front": "front-left", "right front": "front-right",
        "left back": "back-left", "right back": "back-right"
    }
    
    for k, v in compound_map.items():
        if k in text_lower: return v
        
    simples = ["front", "back", "left", "right"]
    for s in simples:
        if s in text_lower: return s
        
    return "NONE" # 매칭 실패 시

def run_debug():
    print(f"🤖 Loading Model: {MODEL_PATH}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"📂 Loading Data...")
    with open(TEST_FILE, 'r') as f:
        dataset = json.load(f)

    print(f"📝 Writing logs to: {LOG_FILE}")
    
    with open(LOG_FILE, "w", encoding="utf-8") as log_f:
        log_f.write("=== Direction Task Raw Logs ===\n")
        log_f.write("Format: [ID] GT_Char(Dir) vs Pred_Char(Dir) | Result\n\n")
        
        for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
            task_type = item.get('question_type', 'Unknown')
            
            # 방향 관련 Task만 필터링
            if task_type not in TARGET_TASKS:
                continue

            # 이미지 처리
            img_rel_path = item['image_path'][0] if isinstance(item['image_path'], list) else item['image_path']
            full_img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
            if not os.path.exists(full_img_path):
                 if img_rel_path.startswith("ViewSpatial-Bench/"):
                    alt_path = full_img_path.replace("ViewSpatial-Bench/ViewSpatial-Bench/", "ViewSpatial-Bench/")
                    if os.path.exists(alt_path): full_img_path = alt_path

            # 질문/선지 처리
            question_main = item['question']
            raw_choices = item['choices']
            
            if isinstance(raw_choices, list): choices_list = raw_choices
            else:
                try: import ast; choices_list = ast.literal_eval(raw_choices)
                except: choices_list = [raw_choices]

            options_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            formatted_choices = [f"{options_idx[i]}. {c}" for i, c in enumerate(choices_list)]
            question_full = f"{question_main}\n" + "\n".join(formatted_choices) + "\nAnswer with the option letter."
            
            answer_gt_char = item['answer'][0].upper()

            # 추론
            messages = [{"role": "user", "content": [{"type": "image", "image": full_img_path}, {"type": "text", "text": question_full}]}]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=16)
            
            pred_text_raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
            pred_char = pred_text_raw[0].upper() if pred_text_raw else "X"

            # === 상세 로그 기록 ===
            try:
                gt_idx = options_idx.index(answer_gt_char)
                pred_idx = options_idx.index(pred_char) if pred_char in options_idx else -1
                
                # 텍스트 내용
                gt_text = choices_list[gt_idx] if gt_idx < len(choices_list) else "ERR"
                pred_text = choices_list[pred_idx] if (pred_idx >= 0 and pred_idx < len(choices_list)) else "ERR"
                
                # 방향 파싱 결과
                gt_dir = extract_direction_debug(gt_text)
                pred_dir = extract_direction_debug(pred_text)
                
                is_correct = (answer_gt_char == pred_char)
                result_str = "✅ CORRECT" if is_correct else "❌ WRONG"
                
                # 로그 파일에 쓰기
                log_f.write("-" * 50 + "\n")
                log_f.write(f"ID: {idx} | Task: {task_type}\n")
                log_f.write(f"Choices: {choices_list}\n")
                log_f.write(f"GT  : {answer_gt_char} -> \"{gt_text}\" (Parsed: {gt_dir})\n")
                log_f.write(f"Pred: {pred_char} -> \"{pred_text}\" (Parsed: {pred_dir})\n")
                log_f.write(f"Result: {result_str}\n")
                
                # 파싱이 안 된 경우 (NONE) 확인용
                if gt_dir == "NONE" or pred_dir == "NONE":
                    log_f.write("⚠️ PARSING FAILED: Could not extract direction from text.\n")

            except ValueError:
                log_f.write(f"⚠️ ERROR: Index parsing failed. GT={answer_gt_char}, Pred={pred_char}\n")
                continue

    print(f"✅ 로그 저장 완료: {LOG_FILE}")
    print("이 파일의 내용을 확인해서 저에게 복사해 주세요!")

if __name__ == "__main__":
    run_debug()