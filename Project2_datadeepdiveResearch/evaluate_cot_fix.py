import os, re, csv, json, torch, base64
import random, argparse
import numpy as np
from PIL import Image
from random import seed
from tqdm.auto import tqdm
from collections import defaultdict
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, default="data_train_scene_split/test.json")
parser.add_argument("--image_folder", type=str, default="/nas_data2/seungwoo/2/ViewSpatial-Bench")
args = parser.parse_args()

model_path = args.model_path
model_name = model_path.split("/")[-1] or model_path.split("/")[-2]
dataset_path = args.dataset_path
image_root = args.image_folder

prompt_format = "" 

print(f"🚀 Loading CoT Model from: {model_path}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)

def extract_choices(choices_str):
    """
    선택지 문자열에서 (라벨, 텍스트) 매핑을 추출
    예: "A. left\nB. right" -> {'A': 'left', 'B': 'right'}
    """
    if not choices_str: return {}
    
    # 패턴: "A. some text" 또는 "(A) some text"
    choices = {}
    pattern = r"\b([A-D])[\.\)]\s*(.*?)(?=\s+[A-D][\.\)]|$)"
    matches = re.findall(pattern, choices_str, re.DOTALL)
    
    for label, text in matches:
        choices[label] = text.strip().lower()
        
    return choices

def extract_option_smart(full_output, choices_dict):
    """
    1단계: 명시적인 라벨(A, B, C, D) 찾기
    2단계: 텍스트 매칭 (답변에 'left'가 있고 선택지 A가 'left'면 A 선택)
    """
    if not full_output: return None
    full_output_lower = full_output.lower()

    # 1. 명확한 결론 패턴 검색 (Priority 1)
    patterns = [
        r"correct option is ([A-D])",
        r"answer is ([A-D])",
        r"therefore, ([A-D])",
        r"option ([A-D])"
    ]
    for p in patterns:
        match = re.search(p, full_output, re.IGNORECASE)
        if match: return match.group(1).upper()

    # 2. 텍스트 매칭 (Semantic Matching)
    # 모델이 "The bird is facing left." 라고 했을 때, 선택지에 'left'가 있는지 확인
    found_labels = []
    for label, text in choices_dict.items():
        # 정확한 매칭을 위해 단어 경계(\b) 사용 고려, 하지만 복합어(front-left) 때문에 단순 포함 확인
        if text in full_output_lower:
            found_labels.append(label)
    
    # 만약 매칭된 텍스트가 하나뿐이라면 그걸 정답으로 간주
    if len(found_labels) == 1:
        return found_labels[0]
    
    # 3. 최후의 수단: 문장 마지막에 등장하는 알파벳
    matches = re.findall(r"\b([A-D])\b", full_output)
    if matches:
        return matches[-1].upper()
        
    return None

def url_to_base64(url):
    full_path = os.path.join(image_root, url)
    if not os.path.exists(full_path):
        if os.path.exists(url): full_path = url 
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    return False

def get_output(image_paths, question):
    if isinstance(image_paths, str): image_paths = [image_paths]
    image_url = [url_to_base64(img) for img in image_paths if url_to_base64(img)]
    if not image_url: return "C"

    content = [{"type": "image", "image": path, "resized_height": 280, "resized_width": 420} for path in image_url]
    messages = [{"role": "user", "content": [*content, {"type": "text", "text": question}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

def evaluate_vlm():
    print(f"Evaluating on: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_questions = 0

    output_path = f"result/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    result_file = f"{output_path}/result_real_cot_fixed.csv"
    
    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Question_Type", "Full_Output", "Parsed_Answer", "Correct_Answer", "IsCorrect"])

        for i, item in enumerate(tqdm(benchmark_data)):
            try:
                image_path = item['image_path']
                # Choices 정보를 파싱하기 위해 원본 question 사용
                choices_str = item.get("choices", "")
                question = item["question"] + choices_str + prompt_format
                correct_answer = item["answer"]
                question_type = item["question_type"]
                
                # 선택지 딕셔너리 추출 (예: {'A': 'left', 'B': 'right'})
                choices_dict = extract_choices(choices_str)
                
                full_output = get_output(image_path, question)
                
                # [수정] 스마트 파싱 함수 사용
                parsed_pred = extract_option_smart(full_output, choices_dict)
                
                # 정답 파싱
                parsed_gt = None
                match_gt = re.search(r"^([A-D])", correct_answer.strip())
                if match_gt:
                    parsed_gt = match_gt.group(1)
                else:
                    parsed_gt = correct_answer.strip()[0] # Fallback

                is_correct = (parsed_pred == parsed_gt) if parsed_pred else False
                
                stats[question_type]["total"] += 1
                total_questions += 1
                if is_correct:
                    stats[question_type]["correct"] += 1
                    total_correct += 1
                    
                writer.writerow([i, question_type, full_output, parsed_pred, correct_answer, is_correct])
                
            except Exception as e:
                print(f"Error on item {i}: {e}")
                continue

    print("\nBenchmark Evaluation Results (Real CoT - Fixed Parsing):")
    print("-" * 60)
    for qtype, values in stats.items():
        if values["total"] > 0:
            print(f"{qtype}: {values['correct']}/{values['total']} = {values['correct']/values['total']:.2%}")
    print("-" * 60)
    if total_questions > 0:
        print(f"Total Accuracy: {total_correct/total_questions:.2%} ({total_correct}/{total_questions})")
    print(f"Result saved to {result_file}")

if __name__ == '__main__':
    evaluate_vlm()