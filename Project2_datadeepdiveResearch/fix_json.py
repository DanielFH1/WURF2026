import json

input_file = 'data_train_scene_split/val_visual_prompt.json'
output_file = 'data_train_scene_split/val_visual_prompt_fixed.json'

with open(input_file, 'r') as f:
    data = json.load(f)

fixed_data = []
for item in data:
    new_item = item.copy()
    
    # 1. 이미지 경로 추출 ('image_path' 키 생성)
    if 'messages' in item:
        # messages 안에서 이미지 찾기
        for msg in item['messages']:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'image':
                        new_item['image_path'] = content['image']
                        break
    elif 'image' in item:
        new_item['image_path'] = item['image']
        
    # 2. 질문 추출 (messages 안에 있으면 꺼내오기)
    if 'messages' in item and 'question' not in new_item:
         for msg in item['messages']:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'text':
                         # 질문에서 보기(A, B, C...) 제거하고 순수 질문만 남기는 로직 필요할 수 있음
                         # 하지만 일단 텍스트 전체를 넣음
                         new_item['question'] = content['text']

    fixed_data.append(new_item)

with open(output_file, 'w') as f:
    json.dump(fixed_data, f, indent=2)

print(f'✅ Fixed JSON saved to: {output_file}')