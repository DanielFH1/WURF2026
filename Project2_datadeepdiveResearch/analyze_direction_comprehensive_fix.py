import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 설정 =================
# GPU 3번 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "./checkpoints/mvsm_baseline_merged"
TEST_FILE = "data_train_scene_split/test.json"
BASE_IMAGE_DIR = "/nas_data2/seungwoo/2/ViewSpatial-Bench"
RESULT_DIR = "data_divedive_results/direction_analysis"
os.makedirs(RESULT_DIR, exist_ok=True)

# 8방향 정의
LABELS = [
    "front", "front-right", "right", "back-right", 
    "back", "back-left", "left", "front-left"
]
# ========================================

def parse_choices(raw_choices):
    """선지 파싱 (이전과 동일)"""
    if isinstance(raw_choices, list):
        if len(raw_choices) == 1 and isinstance(raw_choices[0], str) and '\n' in raw_choices[0]:
            return raw_choices[0].split('\n')
        return raw_choices
    if isinstance(raw_choices, str):
        if '\n' in raw_choices: return raw_choices.split('\n')
        try:
            import ast
            return ast.literal_eval(raw_choices)
        except:
            return [raw_choices]
    return raw_choices

def extract_direction(text):
    """방향 키워드 추출 (이전과 동일)"""
    text = text.lower().replace("-", " ").strip()
    compound_map = {
        "front left": "front-left", "front right": "front-right",
        "back left": "back-left", "back right": "back-right",
        "left front": "front-left", "right front": "front-right",
        "left back": "back-left", "right back": "back-right"
    }
    for k, v in compound_map.items():
        if k in text: return v
    simples = ["front", "back", "left", "right"]
    for s in simples:
        if s in text: return s
    return "other"

def get_direction_components(direction):
    """방향을 구성 요소로 분해 (예: 'front-left' -> {'front', 'left'})"""
    return set(direction.split('-'))

def analyze_error_type(gt, pred):
    if gt == pred: return "Correct"
    
    gt_set = get_direction_components(gt)
    pred_set = get_direction_components(pred)
    
    # 1. Horizontal Flip (좌우 반전) - 가장 치명적
    # (Left <-> Right 가 포함되어 있는지 확인)
    is_hor_flip = False
    if ('left' in gt_set and 'right' in pred_set) or ('right' in gt_set and 'left' in pred_set):
        # 나머지 요소가 같아야 진정한 Flip (예: front-left <-> front-right)
        # 하지만 단순히 left <-> right만 바뀌어도 넓은 의미의 Horizontal Error로 봄
        is_hor_flip = True
    
    if is_hor_flip:
        return "Horizontal_Flip"

    # 2. Vertical Flip (상하/앞뒤 반전)
    is_ver_flip = False
    if ('front' in gt_set and 'back' in pred_set) or ('back' in gt_set and 'front' in pred_set):
        is_ver_flip = True
        
    if is_ver_flip:
        return "Vertical_Flip"

    # 3. Adjacency/Granularity (인접 오류) - 덜 치명적
    # 예: "front-left" vs "front" (교집합이 존재함)
    if not gt_set.isdisjoint(pred_set):
        return "Adjacent_Error"

    # 4. Orthogonal (직각 오류)
    # 교집합도 없고 반대도 아님 (예: front vs left)
    return "Orthogonal_Error"

def run_analysis():
    print(f"🤖 Loading Model: {MODEL_PATH}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"📂 Loading Test Data...")
    with open(TEST_FILE, 'r') as f:
        dataset = json.load(f)

    target_tasks = [
        "Camera perspective - Relative Direction",
        "Person perspective - Relative Direction",
        "Person perspective - Scene Simulation Relative Direction"
    ]
    
    error_counts = {
        "Horizontal_Flip": 0, 
        "Vertical_Flip": 0, 
        "Adjacent_Error": 0, 
        "Orthogonal_Error": 0,
        "Total_Errors": 0
    }
    
    print("🚀 Analyzing Granular Direction Bias...")
    for item in tqdm(dataset):
        task_type = item.get('question_type', 'Unknown')
        if task_type not in target_tasks: continue

        # 이미지 처리
        img_rel_path = item['image_path'][0] if isinstance(item['image_path'], list) else item['image_path']
        full_img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
        if not os.path.exists(full_img_path):
             if img_rel_path.startswith("ViewSpatial-Bench/"):
                alt_path = full_img_path.replace("ViewSpatial-Bench/ViewSpatial-Bench/", "ViewSpatial-Bench/")
                if os.path.exists(alt_path): full_img_path = alt_path

        # 질문 및 선지 처리
        question_main = item['question']
        choices_list = parse_choices(item['choices'])
        
        options_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        formatted_choices = []
        for i, c in enumerate(choices_list):
            if not re.match(r'^[A-Z]\.', str(c)): formatted_choices.append(f"{options_idx[i]}. {c}")
            else: formatted_choices.append(str(c))

        question_full = f"{question_main}\n" + "\n".join(formatted_choices) + "\nAnswer with the option letter."
        answer_gt_char = item['answer'][0].upper()

        # 추론
        messages = [{"role": "user", "content": [{"type": "image", "image": full_img_path}, {"type": "text", "text": question_full}]}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=16)
        
        pred_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        pred_char = pred_text[0].upper() if pred_text else "X"

        try:
            gt_idx = options_idx.index(answer_gt_char)
            pred_idx = options_idx.index(pred_char)
            
            if gt_idx >= len(choices_list) or pred_idx >= len(choices_list): continue
            
            gt_dir = extract_direction(choices_list[gt_idx])
            pred_dir = extract_direction(choices_list[pred_idx])
            
            if gt_dir in LABELS and pred_dir in LABELS:
                if gt_dir != pred_dir:
                    err_type = analyze_error_type(gt_dir, pred_dir)
                    error_counts[err_type] += 1
                    error_counts["Total_Errors"] += 1
        except ValueError:
            continue

    # === 리포트 작성 ===
    total_err = error_counts["Total_Errors"]
    if total_err == 0: total_err = 1

    with open(os.path.join(RESULT_DIR, "granular_bias_report.txt"), "w") as f:
        f.write("=== Granular Directional Error Analysis ===\n")
        f.write(f"Total Direction Errors: {total_err}\n\n")
        
        stats = []
        for key in ["Horizontal_Flip", "Vertical_Flip", "Adjacent_Error", "Orthogonal_Error"]:
            count = error_counts[key]
            rate = count / total_err * 100
            stats.append((key, count, rate))
            
        # 내림차순 정렬
        stats.sort(key=lambda x: x[1], reverse=True)
        
        f.write("--- Error Breakdown ---\n")
        for key, count, rate in stats:
            desc = ""
            if key == "Horizontal_Flip": desc = "(Left <-> Right)"
            elif key == "Vertical_Flip": desc = "(Front <-> Back)"
            elif key == "Adjacent_Error": desc = "(e.g., Front <-> Front-Left)"
            elif key == "Orthogonal_Error": desc = "(e.g., Front <-> Left)"
            
            f.write(f"{key} {desc}: {count} ({rate:.2f}%)\n")
            
        f.write("\n--- Strategic Implication ---\n")
        hor_cnt = error_counts['Horizontal_Flip']
        ver_cnt = error_counts['Vertical_Flip']
        adj_cnt = error_counts['Adjacent_Error']
        
        if hor_cnt > ver_cnt:
            f.write("1. [Main Action] Horizontal Flip is prevalent compared to Vertical Flip.\n")
            f.write("   -> Justifies 'Horizontal Flip Augmentation'.\n")
        
        if adj_cnt > (total_err * 0.3):
            f.write("2. [Insight] High Adjacent Error indicates 'Fine-grained' issues.\n")
            f.write("   -> The model understands the general sector but fails exact precision.\n")
            f.write("   -> Augmentation helps, but adding more diverse viewpoints is better.\n")

    print(f"✨ 정밀 분석 완료! 결과 리포트: {os.path.join(RESULT_DIR, 'granular_bias_report.txt')}")

if __name__ == "__main__":
    run_analysis()