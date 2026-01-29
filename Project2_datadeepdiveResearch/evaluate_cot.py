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

# [수정] CoT 모델은 Chat Template이 자동으로 Assistant 턴을 넘겨주므로, 
# 불필요한 텍스트("Answer:") 추가는 제거하여 학습 때와 조건을 맞춥니다.
prompt_format = "" 

print(f"🚀 Loading CoT Model from: {model_path}")
# [확인] Qwen2.5 클래스 사용 OK
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)

def extract_option_cot(text):
    """
    CoT 모델의 긴 답변에서 진짜 정답을 찾아내는 똑똑한 파싱 함수
    """
    if not text: return None
    
    # 1. 명확한 결론 패턴 검색 ("Therefore, the correct option is A")
    patterns = [
        r"correct option is ([A-D])",
        r"answer is ([A-D])",
        r"Option ([A-D])",
        r"Therefore, ([A-D])",
        r"Therefore, the answer is ([A-D])"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match: return match.group(1).upper()

    # 2. 패턴이 없으면, 문장의 마지막에 등장하는 A-D를 정답으로 간주
    matches = re.findall(r"\b([A-D])\b", text)
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
    
    image_url = [url_to_base64(img) for img in image_paths]
    image_url = [img for img in image_url if img is not False]
    
    # 이미지가 없으면 기본 처리
    if not image_url: return "C"

    content = [{"type": "image", "image": path, "resized_height": 280, "resized_width": 420} for path in image_url]
    
    messages = [{
        "role": "user",
        "content": [*content, {"type": "text", "text": question}]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    
    # [확인] CoT 생성을 위해 충분한 토큰 수 (512) 확보 OK
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    return output_text

def evaluate_vlm():
    print(f"Evaluating on: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_questions = 0

    output_path = f"result/{model_name}"
    os.makedirs(output_path, exist_ok=True)
    result_file = f"{output_path}/result_real_cot.csv"
    
    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Question_Type", "Full_Output", "Parsed_Answer", "Correct_Answer", "IsCorrect"])

        for i, item in enumerate(tqdm(benchmark_data)):
            try:
                image_path = item['image_path']
                # prompt_format은 제거했으므로 순수 질문만 들어감
                question = item["question"] + item.get("choices", "") + prompt_format
                correct_answer = item["answer"]
                question_type = item["question_type"]
                
                full_output = get_output(image_path, question)
                parsed_pred = extract_option_cot(full_output)
                
                # 정답 비교 (문자열 비교)
                # 정답(correct_answer)도 "A. Left" 형태일 수 있으므로 파싱 필요
                parsed_gt = extract_option_cot(correct_answer)
                if not parsed_gt: # 정답이 그냥 "A"인 경우
                    parsed_gt = correct_answer.strip().upper()[0]

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

    print("\nBenchmark Evaluation Results (Real CoT):")
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