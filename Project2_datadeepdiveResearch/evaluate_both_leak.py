import torch
import json
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# ================= 설정 =================
MODEL_PATH = "./checkpoints/mvsm_lora_64_merged"
DATA_DIR = "data_train_old"
FILES = {
    "Validation (Leaked?)": os.path.join(DATA_DIR, "val.jsonl"),
    "Test (Leaked?)": os.path.join(DATA_DIR, "test_hidden.json")
}
# ========================================

def load_data(file_path):
    data = []
    print(f"📂 Loading {file_path}...")
    
    with open(file_path, 'r') as f:
        # JSONL (Validation)
        if file_path.endswith('.jsonl'):
            for line in f:
                item = json.loads(line)
                # ChatML 포맷 -> (Image, Question, Answer) 변환
                img_path = None
                question = None
                answer = None
                
                # Image & Question 추출
                for msg in item['messages']:
                    if msg['role'] == 'user':
                        for content in msg['content']:
                            if content['type'] == 'image':
                                img_path = content['image']
                            elif content['type'] == 'text':
                                question = content['text']
                    elif msg['role'] == 'assistant':
                        for content in msg['content']:
                            answer = content['text']
                            
                if img_path and question and answer:
                    data.append({"image": img_path, "question": question, "answer": answer})

        # JSON List (Test)
        else:
            raw_data = json.load(f)
            for item in raw_data:
                # Raw 포맷 -> (Image, Question, Answer) 변환
                img_path = item['image_path'][0] if isinstance(item['image_path'], list) else item['image_path']
                question = f"{item['question']}\n{item['choices']}\nAnswer with the option letter."
                answer = item['answer']
                data.append({"image": img_path, "question": question, "answer": answer})
                
    return data # 시간 관계상 각 100개만 샘플링 테스트 (전체 돌리려면 제거)

def evaluate(model, processor, dataset, name):
    print(f"\n🚀 Evaluating [{name}] - {len(dataset)} samples")
    correct = 0
    
    for item in tqdm(dataset):
        # 1. Prepare Inputs
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": item['image']},
                {"type": "text", "text": item['question']}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # 2. Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16)
            
        # 3. Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        # 4. Check Answer (첫 글자만 비교)
        pred = output_text[0].upper() if output_text else ""
        gt = item['answer'][0].upper() if item['answer'] else ""
        
        if pred == gt:
            correct += 1
            
    acc = (correct / len(dataset)) * 100
    print(f"📊 Result [{name}]: Accuracy = {acc:.2f}%")
    return acc

def main():
    print(f"🤖 Loading Model from {MODEL_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    for name, path in FILES.items():
        if os.path.exists(path):
            dataset = load_data(path)
            evaluate(model, processor, dataset, name)
        else:
            print(f"⚠️ File not found: {path}")

if __name__ == "__main__":
    main()