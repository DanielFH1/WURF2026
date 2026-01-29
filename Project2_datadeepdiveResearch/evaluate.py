# evaluate.py (수정 버전)
import os, re, csv, json, torch, base64
import random, argparse
import numpy as np
from PIL import Image
from random import seed
from openai import OpenAI
from tqdm.auto import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoProcessor
# Llama-3.2-11B-Vision
from transformers import MllamaForConditionalGeneration
# Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration
# Qwen2.5-VL
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration
# LlavaOnevision
from transformers import LlavaOnevisionForConditionalGeneration
# Intern2.5/3
# from lmdeploy.vl import load_image
# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
# LlavaNextVideo
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

seed(1234)
np.random.seed(1234)

# ★ 수정된 부분: 인자 추가
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="gpt-4o")
parser.add_argument("--dataset_path", type=str, default="eval/ViewSpatial-Bench.json", help="Path to the benchmark JSON file")
parser.add_argument("--image_folder", type=str, default=".", help="Root folder for images")
args = parser.parse_args()

model_path = args.model_path
model_name = model_path.split("/")[-1]
# 모델 이름이 경로일 경우(./checkpoints/...), 폴더 이름만 따오기
if model_name == "":
    model_name = model_path.split("/")[-2]

dataset_path = args.dataset_path
image_root = args.image_folder

prompt_format = "\nReply only to the corresponding option.\nAnswer:"

# Set the size of the incoming image for qwen
min_pixels = 256*28*28
max_pixels = 1280*28*28

# Set up the model
if model_name == 'gemini-2.0-flash-001':
    API_KEY = ""  # your api key
    base_url = ""  # Change to your own base_url
    client = OpenAI(api_key=API_KEY, base_url=base_url)
    print(f"Model gemini-2.0-flash series:{model_name} is running!")
    
elif model_name == 'gpt-4o':
    client = OpenAI(api_key="")  # your api key
    client.base_url = ""  # Change to your own base_url
    print(f"Model gpt-4o series:{model_name} is running!")

# ★ MVSM (Qwen2.5 기반) 로드 로직 추가 (mvsm_merged 등)
elif "Qwen2.5-VL" in model_path or "mvsm" in model_path.lower():
    # 로컬 경로 모델 로딩 지원
    print(f"Loading Qwen2.5/MVSM model from: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    print(f"Model Qwen2.5-VL series:{model_name} is running!")

elif "Qwen2-VL" in model_name :
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    print(f"Model Qwen2-VL series:{model_name} is running!")

# ... (나머지 모델 생략, 필요시 추가) ...
else:
    # 혹시 모를 Fallback
    if "Qwen" in model_path: # 이름 파싱 실패 시 강제 로드
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    else:
        model = None
        processor = None

def extract_option(text):
    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None

def url_to_base64(url):
    # 이미지 경로 결합 (root + relative_path)
    full_path = os.path.join(image_root, url)
    
    # 중복 경로 방지 (데이터셋 경로가 이미 ViewSpatial-Bench를 포함하는 경우)
    if not os.path.exists(full_path):
        # 혹시 image_folder를 안 썼을 때를 대비해 원본 url도 체크
        if os.path.exists(url):
            full_path = url
    
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    else:
        print(f"이미지 {full_path} 가 존재하지 않습니다!")
        return False

def get_output(image_path, question):
    image_url = [url_to_base64(image) for image in image_path]
    # base64 변환 실패 시 처리
    image_url = [img for img in image_url if img is not False]
    
    if not image_url:
        return "C" # 이미지 없으면 찍기 (Failover)

    # Qwen2.5-VL / MVSM
    if "Qwen2.5-VL" in model_path or "mvsm" in model_path.lower() or "Qwen" in model_path:
        content = [{"type": "image", "image": path,"resized_height": 280,"resized_width": 420} for path in image_url]

        messages = [
            {
                "role": "user",
                "content": [
                    *content,
                    {
                        "type": "text",
                        "text": question
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        pred = str(output_text[0])
        
    else:
        pred = ''
        
    return pred

def evaluate_vlm(benchmark_file):
    print(f"Evaluating on: {benchmark_file}")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_questions = 0

    output_path = f"result/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    result_file = f"{output_path}/result_{model_name}.csv"
    
    # 진행 상황 확인을 위해 tqdm 사용
    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Question", "Question_Type", "Predicted Answer", "Correct Answer", "IsCorrect"])

        for i, item in enumerate(tqdm(benchmark_data)):
            try:
                image_path = item['image_path']
                question = item["question"] + item["choices"] + prompt_format
                correct_answer = item["answer"]
                question_type = item["question_type"]
                stats[question_type]["total"] += 1
                total_questions += 1
                
                predicted_answer = get_output(image_path, question)
                predicted_answer_ = predicted_answer.split("\n")[-1]
                is_correct = extract_option(predicted_answer_) == extract_option(correct_answer)
                
                if is_correct:
                    stats[question_type]["correct"] += 1
                    total_correct += 1
                writer.writerow([i, question, question_type, predicted_answer, correct_answer, is_correct])
            except Exception as e:
                print(f"Error on item {i}: {e}")
                continue

    print("Benchmark Evaluation Results:")
    print("----------------------------------------------------------")
    for qtype, values in stats.items():
        correct = values["correct"]
        total = values["total"]
        if total > 0:
            accuracy = correct / total
            print(f"{qtype}: {correct}/{total} = {accuracy:.2%}")
    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print("----------------------------------------------------------")
        print(f"Accuracy of {model_name}: {overall_accuracy:.2%} ({total_correct}/{total_questions})")
        print("----------------------------------------------------------")
    print(f"Result saved to {result_file}")

if __name__ == '__main__':
    # 인자로 받은 경로 사용
    evaluate_vlm(dataset_path)