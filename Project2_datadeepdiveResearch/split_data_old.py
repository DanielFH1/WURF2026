import json
import random
import os

# 1. 데이터 로드
input_file = "eval/ViewSpatial-Bench.json"
with open(input_file, "r") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# 2. 데이터 셔플
random.seed(42)
random.shuffle(data)

# 3. 데이터 3분할 (Train 80% / Val 10% / Test 10%)
n_total = len(data)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
# 나머지는 test

train_data = data[:n_train]
val_data = data[n_train : n_train + n_val]
test_data = data[n_train + n_val:] # <-- 얘는 학습때 절대 안 보여줍니다.

print(f"Train samples: {len(train_data)}")
print(f"Val samples:   {len(val_data)}")
print(f"Test samples:  {len(test_data)} (Hidden for final evaluation)")

# 4. Qwen 포맷 변환 함수 (이전과 동일)
def format_for_qwen(items, image_root_dir):
    formatted = []
    for item in items:
        img_paths = item['image_path']
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        
        valid_img_paths = []
        for p in img_paths:
            full_path = os.path.join(image_root_dir, p)
            if os.path.exists(p):
                valid_img_paths.append(p)
            elif os.path.exists(os.path.join(image_root_dir, p)):
                valid_img_paths.append(os.path.join(image_root_dir, p))
        
        if not valid_img_paths:
            continue 

        # Qwen 학습 포맷 (messages 구조)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": valid_img_paths[0]}, 
                    {"type": "text", "text": item["question"] + "\n" + item["choices"] + "\nAnswer:"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["answer"]}] 
            }
        ]
        formatted.append({"messages": conversation})
    return formatted

# 5. 저장
os.makedirs("data_train_old", exist_ok=True)

# Train과 Val은 학습용 포맷(.jsonl)으로 저장 (messages 구조)
train_formatted = format_for_qwen(train_data, ".")
val_formatted = format_for_qwen(val_data, ".")

with open("data_train_old/train.jsonl", "w") as f:
    for entry in train_formatted:
        json.dump(entry, f); f.write('\n')

with open("data_train_old/val.jsonl", "w") as f:
    for entry in val_formatted:
        json.dump(entry, f); f.write('\n')

# ★중요★ Test 데이터는 나중에 evaluate.py로 돌려야 하므로
# 학습용 포맷이 아니라 '원본 벤치마크 JSON 포맷' 그대로 저장합니다.
with open("data_train_old/test_hidden.json", "w") as f:
    json.dump(test_data, f, indent=4)

print("\n✅ Data preparation complete!")
print("1. data_train_old/train.jsonl  (For Training)")
print("2. data_train_old/val.jsonl    (For Validation)")
print("3. data_train_old/test_hidden.json (For Final Testing - 원본 포맷 유지)")