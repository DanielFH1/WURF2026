import torch
import json
import os
from tqdm import tqdm
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 설정 =================
# ★ 방금 Merge한 모델 경로
MODEL_PATH = "./checkpoints/mvsm_epoch10_merged"

# ★ 평가할 데이터 (Scene Split된 Test Set)
TEST_FILE = "data_train_scene_split/test.json"

# ★ 이미지 기본 경로
BASE_IMAGE_DIR = "/nas_data2/seungwoo/2/ViewSpatial-Bench"
# ========================================

def load_test_data():
    print(f"📂 Loading Test Data: {TEST_FILE}")
    with open(TEST_FILE, 'r') as f:
        data = json.load(f)
    print(f"   -> {len(data)} samples loaded.")
    return data

def evaluate():
    # 1. 모델 & 프로세서 로드
    print(f"🤖 Loading Model from {MODEL_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 2. 데이터 로드
    dataset = load_test_data()
    
    # Task별 통계 저장소
    # 구조: {'Task Name': {'correct': 0, 'total': 0}}
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    print("🚀 Start Detailed Evaluation...")
    for item in tqdm(dataset):
        # --- 전처리 (이미지 경로) ---
        img_rel_path = item['image_path'][0] if isinstance(item['image_path'], list) else item['image_path']
        full_img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
        
        # 경로 보정 로직
        if not os.path.exists(full_img_path):
            if img_rel_path.startswith("ViewSpatial-Bench/"):
                alt_path = full_img_path.replace("ViewSpatial-Bench/ViewSpatial-Bench/", "ViewSpatial-Bench/")
                if os.path.exists(alt_path):
                    full_img_path = alt_path

        # --- Task Type 추출 ---
        # 예: "camera_rel_dir", "person_obj_ori" 등
        task_type = item.get('question_type', 'Unknown Task')

        # --- 질문 구성 ---
        question_text = f"{item['question']}\n{item['choices']}\nAnswer with the option letter."
        
        # --- 추론 ---
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": full_img_path},
                {"type": "text", "text": question_text},
            ],
        }]
        
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        # --- 채점 ---
        pred = output_text[0].upper() if output_text else ""
        gt = item['answer'][0].upper()
        
        # 통계 업데이트
        stats[task_type]['total'] += 1
        stats['Overall']['total'] += 1  # 전체 합계
        
        if pred == gt:
            stats[task_type]['correct'] += 1
            stats['Overall']['correct'] += 1

    # 3. 결과 출력 (Task별 리포트)
    print("\n" + "="*60)
    print(f"{'Task Type':<30} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 60)
    
    # Task 이름을 정렬해서 출력
    for task, data in sorted(stats.items()):
        if task == 'Overall': continue # 맨 마지막에 출력하기 위해 스킵
        acc = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
        print(f"{task:<30} | {data['total']:<8} | {data['correct']:<8} | {acc:.2f}%")
        
    print("-" * 60)
    # Overall 출력
    ov = stats['Overall']
    ov_acc = (ov['correct'] / ov['total']) * 100 if ov['total'] > 0 else 0
    print(f"{'Overall Average':<30} | {ov['total']:<8} | {ov['correct']:<8} | {ov_acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    evaluate()