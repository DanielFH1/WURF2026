import os
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import gc # 가비지 컬렉터

# ================= 설정 =================
MODEL_PATH = "./checkpoints/mvsm_visual_cot_merged"
CLEAN_DATA_PATH = "data_train_scene_split/test.json"
BOXED_DATA_PATH = "data_train_scene_split/test_visual_prompt.json"
IMAGE_ROOT = "/nas_data2/seungwoo/2/ViewSpatial-Bench"
OUTPUT_DIR = "result/saliency_maps_gradient"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 메모리 단편화 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# =======================================

def get_actual_image_path(img_path_entry):
    if isinstance(img_path_entry, list): path_str = img_path_entry[0]
    else: path_str = img_path_entry
    full_path_a = os.path.join(IMAGE_ROOT, path_str)
    if os.path.exists(full_path_a): return full_path_a
    if os.path.exists(path_str): return path_str
    parts = path_str.split(os.sep)
    if len(parts) > 1:
        shorter_path = os.path.join(*parts[1:])
        full_path_c = os.path.join(IMAGE_ROOT, shorter_path)
        if os.path.exists(full_path_c): return full_path_c
    return None

def get_saliency_map(model, processor, image_path_entry, text_prompt):
    # 메모리 청소
    torch.cuda.empty_cache()
    gc.collect()

    full_path = get_actual_image_path(image_path_entry)
    if not full_path:
        print(f"❌ Image missing: {image_path_entry}")
        return None, None
    
    image = Image.open(full_path).convert("RGB")
    
    messages = [{"role": "user", "content": [{"type": "image", "image": full_path}, {"type": "text", "text": text_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    
    # 픽셀에 대한 Gradient 추적 설정
    if 'pixel_values' in inputs:
        inputs['pixel_values'].requires_grad_(True)
        inputs['pixel_values'].retain_grad()
    else:
        return None, None

    # Forward Pass
    model.zero_grad()
    
    # [핵심] Gradient Checkpointing이 켜져 있으면 forward 시 메모리를 아낌
    outputs = model(**inputs)
    
    logits = outputs.logits
    # 가장 마지막에 생성될 토큰의 Logit을 타겟으로 잡음
    # (Qwen2-VL은 답변을 생성하기 직전의 상태)
    next_token_logits = logits[0, -1, :]
    target_token_index = next_token_logits.argmax()
    score = next_token_logits[target_token_index]
    
    # Backward Pass
    score.backward()
    
    gradients = inputs['pixel_values'].grad
    if gradients is None:
        print("❌ Gradients are None.")
        return None, None

    # Saliency 계산 (채널 평균)
    saliency = gradients.abs().mean(dim=-1).detach().cpu() # CPU로 바로 내림
    
    # 메모리 해제
    del gradients, outputs, logits, score
    torch.cuda.empty_cache()

    # Grid 복원
    grid_thw = inputs['image_grid_thw'][0]
    h, w = grid_thw[1], grid_thw[2]
    expected_len = h * w
    
    # Qwen2-VL 2x2 Pooling 고려 (visual tokens = h//2 * w//2)
    # 하지만 pixel_values의 길이는 보통 h*w (Before pooling) 이거나 pooling 후일 수 있음.
    # pixel_values.grad의 shape[0] 확인 필요.
    # pixel_values shape은 [Total_Pixels, Channels] (Flattened patches)
    
    # 만약 saliency 길이가 h*w와 같다면:
    if saliency.shape[0] == expected_len:
        heatmap = saliency.view(h, w).float().numpy()
    else:
        # 길이가 안 맞으면 (보통 Pooling 때문)
        # Vision Tokens (h//2 * w//2) 만큼만 뒤에서 자름
        vision_len = (h//2) * (w//2)
        if saliency.shape[0] >= vision_len:
            saliency = saliency[-vision_len:]
            heatmap = saliency.view(h//2, w//2).float().numpy()
        else:
            print(f"⚠️ Shape Mismatch: {saliency.shape} vs {h}x{w}")
            return None, None

    # 리사이징 및 정규화
    img_w, img_h = image.size
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    
    # 노이즈 제거 (상위 1% 클리핑)
    threshold = np.percentile(heatmap, 99)
    heatmap = np.clip(heatmap, 0, threshold)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return image, heatmap

def visualize_comparison(model, processor, clean_item, boxed_item, idx):
    print(f"\n🎨 Processing ID {idx}...")
    
    try:
        img_clean, map_clean = get_saliency_map(model, processor, clean_item['image_path'], clean_item['question'])
        if map_clean is None: 
            print("   ❌ Clean map generation failed.")
            return

        img_boxed, map_boxed = get_saliency_map(model, processor, boxed_item['image_path'], boxed_item['question'])
        if map_boxed is None: 
            print("   ❌ Boxed map generation failed.")
            return

        # 시각화 저장
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_clean)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(img_clean)
        axes[1].imshow(map_clean, alpha=0.6, cmap='jet')
        axes[1].set_title("Original Saliency")
        axes[1].axis('off')
        
        axes[2].imshow(img_boxed)
        axes[2].set_title("Visual Prompt Image")
        axes[2].axis('off')
        
        axes[3].imshow(img_boxed)
        axes[3].imshow(map_boxed, alpha=0.6, cmap='jet')
        axes[3].set_title("Boxed Saliency\n(Tunnel Vision Check)")
        axes[3].axis('off')
        
        save_path = os.path.join(OUTPUT_DIR, f"saliency_map_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) # 메모리 해제
        print(f"   ✅ Saved: {save_path}")
        
    except Exception as e:
        print(f"   ❌ Error processing ID {idx}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 Initializing...")
    
    # [최적화 1] bfloat16 사용 (메모리 절반)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0",
    )
    
    # [최적화 2] Gradient Checkpointing 활성화 (메모리 사용량 대폭 감소)
    model.gradient_checkpointing_enable()
    
    # Checkpointing 사용 시 입력의 Gradients를 켜줘야 함
    model.enable_input_require_grads()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    print("✅ Model loaded with optimizations (bf16 + checkpointing)")

    # 데이터 로드
    with open(CLEAN_DATA_PATH, 'r') as f: clean_data = json.load(f)
    with open(BOXED_DATA_PATH, 'r') as f: boxed_data = json.load(f)
    
    clean_map = {i: item for i, item in enumerate(clean_data)}
    boxed_map = {i: item for i, item in enumerate(boxed_data)}
    
    target_ids = [i for i, item in enumerate(clean_data) if "Scene Simulation" in item.get('question_type', '')]
    target_ids = target_ids[:5] # 5개만 테스트
    
    print(f"🧪 Generating Maps for {len(target_ids)} samples...")
    
    for idx in target_ids:
        if idx in clean_map and idx in boxed_map:
            visualize_comparison(model, processor, clean_map[idx], boxed_map[idx], idx)

    print("✅ All Done!")

if __name__ == "__main__":
    main()