import torch
import json
import os
import matplotlib.pyplot as plt
import textwrap
import re
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 설정 =================
# GPU 3번 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "./checkpoints/mvsm_baseline_merged"
TEST_FILE = "data_train_scene_split/test.json"
BASE_IMAGE_DIR = "/nas_data2/seungwoo/2/ViewSpatial-Bench"

# 결과 저장 경로
RESULT_DIR = "data_divedive_results/experiment_1"
os.makedirs(RESULT_DIR, exist_ok=True)
# ========================================

def get_font(size=40):
    """서버 환경에서 큰 폰트 로드"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "arial.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()

def clean_choice_text(text):
    return re.sub(r'^[A-Z]\.\s*', '', str(text)).strip()

def create_analysis_image(img_path, q_text, choices_list, answer_idx, pred_idx, is_correct, save_path):
    try:
        # 1. 이미지 로드 및 리사이즈 (너무 큰 이미지 방지)
        orig_img = Image.open(img_path).convert("RGB")
        max_img_width = 1400
        if orig_img.width > max_img_width:
            ratio = max_img_width / orig_img.width
            new_height = int(orig_img.height * ratio)
            orig_img = orig_img.resize((max_img_width, new_height), Image.Resampling.LANCZOS)
        
        img_w, img_h = orig_img.size
        
        # 2. 폰트 설정
        font_title = get_font(50)
        font_normal = get_font(36)
        font_small = get_font(32)
        
        margin = 60
        line_spacing = 20
        section_spacing = 40
        
        # 3. 캔버스 너비 결정 (이미지 너비보다 충분히 크게)
        canvas_w = max(img_w + margin * 2, 1600)
        
        # 4. 텍스트 내용 구성
        options_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Result 섹션
        result_str = "✅ CORRECT" if is_correct else "❌ WRONG"
        result_color = (0, 150, 0) if is_correct else (200, 0, 0)
        
        # Question 섹션
        question_lines = textwrap.wrap(q_text, width=int((canvas_w - margin*2) / 20))
        
        # Options 섹션 (각 선지별로 처리)
        formatted_choices = []
        for idx, raw_choice in enumerate(choices_list):
            letter = options_idx[idx]
            clean_text = clean_choice_text(raw_choice)
            
            # 레이블 결정
            labels = []
            if letter == answer_idx:
                labels.append("Ground Truth")
            if letter == pred_idx:
                labels.append("Model Prediction")
            
            # 선지 텍스트 구성
            label_str = f" ({', '.join(labels)})" if labels else ""
            choice_text = f"{letter}. {clean_text}{label_str}"
            
            # 긴 선지는 줄바꿈
            wrapped = textwrap.wrap(choice_text, width=int((canvas_w - margin*2) / 20))
            formatted_choices.extend(wrapped)
        
        # 5. 텍스트 영역 높이 계산
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        # Result 높이
        result_bbox = dummy_draw.textbbox((0, 0), result_str, font=font_title)
        result_h = result_bbox[3] - result_bbox[1]
        
        # Question 높이
        question_h = len(question_lines) * (36 + line_spacing)
        
        # Options 높이
        options_h = len(formatted_choices) * (32 + line_spacing)
        
        # 총 텍스트 영역 높이
        text_area_h = (margin + 
                       result_h + section_spacing + 
                       50 + line_spacing +  # "[Question]" 헤더
                       question_h + section_spacing + 
                       50 + line_spacing +  # "[Options]" 헤더
                       options_h + margin)
        
        # 6. 최종 캔버스 생성
        canvas_h = img_h + text_area_h
        final_img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(final_img)
        
        # 7. 이미지 붙이기 (중앙 정렬, 비율 유지)
        img_x = (canvas_w - img_w) // 2
        final_img.paste(orig_img, (img_x, 0))
        
        # 8. 텍스트 그리기
        curr_y = img_h + margin
        
        # Result
        draw.text((margin, curr_y), result_str, fill=result_color, font=font_title)
        curr_y += result_h + section_spacing
        
        # Question 헤더
        draw.text((margin, curr_y), "[Question]", fill=(0, 0, 0), font=font_normal)
        curr_y += 50 + line_spacing
        
        # Question 내용
        for line in question_lines:
            draw.text((margin, curr_y), line, fill=(40, 40, 40), font=font_small)
            curr_y += 32 + line_spacing
        
        curr_y += section_spacing
        
        # Options 헤더
        draw.text((margin, curr_y), "[Options]", fill=(0, 0, 0), font=font_normal)
        curr_y += 50 + line_spacing
        
        # Options 내용 (강조 표시)
        for choice_line in formatted_choices:
            # Ground Truth나 Prediction이 포함된 줄은 배경색 추가
            text_color = (40, 40, 40)
            if "Ground Truth" in choice_line or "Model Prediction" in choice_line:
                # 배경 박스 그리기
                bbox = draw.textbbox((margin, curr_y), choice_line, font=font_small)
                draw.rectangle(
                    [(bbox[0]-5, bbox[1]-2), (bbox[2]+5, bbox[3]+2)],
                    fill=(255, 255, 200) if "Ground Truth" in choice_line else (230, 230, 255)
                )
                text_color = (0, 0, 0)
            
            draw.text((margin, curr_y), choice_line, fill=text_color, font=font_small)
            curr_y += 32 + line_spacing

        # 9. 저장
        final_img.save(save_path, quality=95)
        
    except Exception as e:
        print(f"⚠️ 이미지 저장 실패 ({save_path}): {e}")

def run_analysis():
    print(f"🤖 Loading Model: {MODEL_PATH}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"📂 Loading Data: {TEST_FILE}")
    with open(TEST_FILE, 'r') as f:
        dataset = json.load(f)

    stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': []})
    
    print("🚀 Deep Dive Visual Analysis (Top-Bottom Layout)...")
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        # 이미지 경로 처리
        img_rel_path = item['image_path'][0] if isinstance(item['image_path'], list) else item['image_path']
        full_img_path = os.path.join(BASE_IMAGE_DIR, img_rel_path)
        
        if not os.path.exists(full_img_path):
             if img_rel_path.startswith("ViewSpatial-Bench/"):
                alt_path = full_img_path.replace("ViewSpatial-Bench/ViewSpatial-Bench/", "ViewSpatial-Bench/")
                if os.path.exists(alt_path):
                    full_img_path = alt_path

        # 데이터 파싱
        task_type = item.get('question_type', 'Unknown')
        question_main = item['question']
        raw_choices = item['choices']
        
        # 선지 리스트 확보
        formatted_choices = []
        options_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        if isinstance(raw_choices, list):
            choices_list_pure = raw_choices
            for i, c in enumerate(raw_choices): formatted_choices.append(f"{options_idx[i]}. {c}")
        else:
            try:
                import ast
                choices_list_pure = ast.literal_eval(raw_choices)
                for i, c in enumerate(choices_list_pure): formatted_choices.append(f"{options_idx[i]}. {c}")
            except:
                choices_list_pure = [raw_choices]
                formatted_choices.append(raw_choices)

        choices_str = "\n".join(formatted_choices)
        question_full = f"{question_main}\n{choices_str}\nAnswer with the option letter."
        answer_gt = item['answer'][0].upper()

        # 추론
        messages = [{"role": "user", "content": [{"type": "image", "image": full_img_path}, {"type": "text", "text": question_full}]}]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=16)
        
        pred_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        pred_char = pred_text[0].upper() if pred_text else "X"

        # 통계
        is_correct = (pred_char == answer_gt)
        stats[task_type]['total'] += 1
        if is_correct:
            stats[task_type]['correct'] += 1
        else:
            stats[task_type]['errors'].append((answer_gt, pred_char, choices_list_pure))

        # 저장
        status_folder = "Correct" if is_correct else "Incorrect"
        save_dir = os.path.join(RESULT_DIR, task_type, status_folder)
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{idx:04d}_GT-{answer_gt}_Pred-{pred_char}.jpg"
        
        # 개선된 레이아웃 함수 호출
        create_analysis_image(
            full_img_path, 
            question_main, 
            choices_list_pure, 
            answer_gt, 
            pred_char, 
            is_correct, 
            os.path.join(save_dir, filename)
        )

    # 차트 및 리포트 생성
    generate_report(stats)

def generate_report(stats):
    print("📊 Generating Final Report...")
    tasks = sorted(stats.keys())
    accuracies = []
    labels = []
    
    for t in tasks:
        correct = stats[t]['correct']
        total = stats[t]['total']
        acc = (correct / total) * 100 if total > 0 else 0
        accuracies.append(acc)
        labels.append(f"{acc:.1f}% ({correct}/{total})")
        
    plt.figure(figsize=(14, 8))
    bars = plt.barh(tasks, accuracies, color='#4A90E2', alpha=0.8)
    plt.xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Task-wise Accuracy (Baseline)', fontsize=16, fontweight='bold')
    plt.xlim(0, 115)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    for bar, label in zip(bars, labels):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, label, 
                 va='center', fontsize=12, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "accuracy_chart.svg"), format='svg')
    plt.savefig(os.path.join(RESULT_DIR, "accuracy_chart.png"), dpi=300)
    
    with open(os.path.join(RESULT_DIR, "error_analysis.txt"), "w") as f:
        f.write("=== Deep Dive Error Analysis ===\n\n")
        total_correct = sum(s['correct'] for s in stats.values())
        total_cnt = sum(s['total'] for s in stats.values())
        if total_cnt > 0:
            f.write(f"Overall Accuracy: {total_correct/total_cnt*100:.2f}% ({total_correct}/{total_cnt})\n\n")
        
        for task in tasks:
            f.write(f"## Task: {task}\n")
            errs = stats[task]['errors']
            f.write(f"  - Accuracy: {stats[task]['correct']/stats[task]['total']*100:.2f}%\n")
            
            semantic_patterns = defaultdict(int)
            options_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for gt_char, pred_char, choices in errs:
                try:
                    gt_idx = options_idx.index(gt_char)
                    pred_idx = options_idx.index(pred_char)
                    gt_text = choices[gt_idx] if gt_idx < len(choices) else "Unknown"
                    pred_text = choices[pred_idx] if pred_idx < len(choices) else "Unknown"
                    semantic_patterns[f"'{gt_text}' -> '{pred_text}'"] += 1
                except:
                    semantic_patterns[f"{gt_char} -> {pred_char}"] += 1

            f.write("  - Top Confusion Patterns:\n")
            for p, c in sorted(semantic_patterns.items(), key=lambda x:x[1], reverse=True)[:10]:
                f.write(f"    {p}: {c} times\n")
            f.write("\n")
            
    print(f"✅ Deep Dive 완료! 결과 폴더: {RESULT_DIR}")

if __name__ == "__main__":
    run_analysis()