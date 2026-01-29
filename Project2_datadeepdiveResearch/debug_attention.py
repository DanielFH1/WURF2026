import os
import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 설정 =================
MODEL_PATH = "./checkpoints/mvsm_visual_cot_merged"
# 아까 에러 로그에 떴던 그 이미지 경로를 그대로 씁니다.
TEST_IMAGE_PATH = "/nas_data2/seungwoo/2/ViewSpatial-Bench/ViewSpatial-Bench/scannetv2_val/scene0651_01/original_images/20.jpg"
TEXT_PROMPT = "Describe this image."
# =======================================

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def recursive_hook(name):
    def hook(module, input, output):
        print(f"🪝 Hook triggered for: {name}")
        if isinstance(output, tuple):
            print(f"   Output is tuple with length: {len(output)}")
            for i, item in enumerate(output):
                if item is None:
                    print(f"     [{i}] None")
                elif isinstance(item, torch.Tensor):
                    print(f"     [{i}] Tensor shape: {item.shape}")
                else:
                    print(f"     [{i}] Type: {type(item)}")
        elif isinstance(output, torch.Tensor):
             print(f"   Output is Tensor shape: {output.shape}")
        else:
             print(f"   Output type: {type(output)}")
    return hook

def main():
    print_separator("LOADING MODEL")
    try:
        # Flash Attention을 끄고(eager) 로드
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float32, 
            device_map="cuda:0",
            attn_implementation="eager" 
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    print_separator("CONFIG INSPECTION")
    # 설정이 제대로 먹혔는지 확인
    print(f"Model Config 'output_attentions': {getattr(model.config, 'output_attentions', 'Not Set')}")
    print(f"Model Config 'attn_implementation': {getattr(model.config, 'attn_implementation', 'Not Set')}")
    
    if hasattr(model, 'visual'):
        print(f"Visual Config 'output_attentions': {getattr(model.visual.config, 'output_attentions', 'Not Set')}")
    else:
        print("❌ 'model.visual' attribute not found!")

    print_separator("STRUCTURE INSPECTION")
    # 실제 어텐션 모듈이 어떤 클래스인지 확인 (FlashAttention인지 아닌지)
    if hasattr(model, 'visual') and hasattr(model.visual, 'blocks'):
        first_block = model.visual.blocks[0]
        if hasattr(first_block, 'attn'):
            print(f"Attention Module type: {type(first_block.attn)}")
        else:
            print("❌ 'attn' module not found in visual block.")
    
    print_separator("HOOK REGISTRATION")
    # Vision Encoder의 마지막 블록에 Hook 걸기
    if hasattr(model, 'visual') and hasattr(model.visual, 'blocks'):
        target_layer = model.visual.blocks[-1].attn
        print(f"Registering hook on: model.visual.blocks[-1].attn")
        target_layer.register_forward_hook(recursive_hook("Visual.Block[-1].Attn"))
    
    print_separator("INFERENCE TEST")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"⚠️ Test image not found at {TEST_IMAGE_PATH}. (Cannot proceed with inference)")
        return

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": TEST_IMAGE_PATH},
            {"type": "text", "text": TEXT_PROMPT}
        ]}
    ]
    
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        print("Running forward pass with output_attentions=True...")
        
        # 1. 전체 모델 호출 테스트
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        print(f"\n[Full Model Output Keys]: {outputs.keys()}")
        if hasattr(outputs, 'vision_attentions') and outputs.vision_attentions is not None:
             print(f"✅ outputs.vision_attentions found! Length: {len(outputs.vision_attentions)}")
        else:
             print("❌ outputs.vision_attentions is Missing.")

        print_separator("DIRECT VISUAL ENCODER TEST")
        
        # 2. Vision Encoder 직접 호출 테스트
        if 'pixel_values' in inputs:
            print("Calling model.visual directly...")
            pixel_values = inputs['pixel_values'].to(model.dtype)
            grid_thw = inputs['image_grid_thw']
            
            with torch.no_grad():
                # 여기서 Config 강제 업데이트
                model.visual.config.output_attentions = True
                
                # 직접 호출
                visual_outputs = model.visual(
                    hidden_states=pixel_values,
                    grid_thw=grid_thw,
                    output_attentions=True
                )
            
            print(f"\n[Visual Output Type]: {type(visual_outputs)}")
            
            # 튜플인지 객체인지 확인
            if isinstance(visual_outputs, tuple):
                print(f"Visual Output is Tuple with length: {len(visual_outputs)}")
                for i, item in enumerate(visual_outputs):
                    if isinstance(item, torch.Tensor):
                        print(f"  [{i}] Tensor shape: {item.shape}")
                    elif isinstance(item, tuple):
                        print(f"  [{i}] Tuple (Attentions?) length: {len(item)}")
                        if len(item) > 0 and isinstance(item[0], torch.Tensor):
                            print(f"      Element 0 shape: {item[0].shape}")
            elif hasattr(visual_outputs, 'attentions'):
                 print(f"Visual Output object has 'attentions': {visual_outputs.attentions is not None}")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

    print_separator("DONE")

if __name__ == "__main__":
    main()