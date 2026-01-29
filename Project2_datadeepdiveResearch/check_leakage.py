import json
import os
import re

# ================= 경로 설정 =================
BASE_DIR = "/nas_data2/seungwoo/2/ViewSpatial-Bench/data_train_old"

TRAIN_FILE = os.path.join(BASE_DIR, "train.jsonl")
VAL_FILE = os.path.join(BASE_DIR, "val.jsonl")
TEST_FILE = os.path.join(BASE_DIR, "test_hidden.json")
# ============================================

def extract_path_from_chatml(item):
    """Train/Val 데이터(ChatML 포맷)에서 이미지 경로 추출"""
    try:
        # messages -> content 순회
        if 'messages' in item:
            for msg in item['messages']:
                if msg['role'] == 'user':
                    for content in msg['content']:
                        if content.get('type') == 'image':
                            return content.get('image')
    except:
        pass
    return None

def extract_path_from_raw(item):
    """Test 데이터(Raw 포맷)에서 이미지 경로 추출"""
    try:
        # image_path는 리스트 형태임
        path_list = item.get('image_path')
        if path_list and isinstance(path_list, list):
            return path_list[0]
        elif isinstance(path_list, str):
            return path_list
    except:
        pass
    return None

def get_scene_id(path):
    """이미지 경로에서 Scene ID (sceneXXXX_XX) 추출"""
    if not path:
        return None
    # 정규식으로 sceneID 찾기
    match = re.search(r'(scene\d+_\d+)', path)
    if match:
        return match.group(1)
    return None

def check_leakage():
    print("📂 데이터 정밀 분석 시작...")
    
    # 1. Train 데이터 로드 및 Scene ID 수집
    print("   - Train 데이터 로딩 및 분석 중...")
    train_scenes = set()
    scannet_train_count = 0
    
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            path = extract_path_from_chatml(item)
            scene_id = get_scene_id(path)
            
            if scene_id:
                train_scenes.add(scene_id)
                scannet_train_count += 1
                
    print(f"   => Train 내 ScanNet 데이터: {scannet_train_count}개")
    print(f"   => 학습한 고유 장소(Scene) 수: {len(train_scenes)}개")
    print("-" * 50)

    # 2. 누수 검사 함수
    def analyze_split(name, file_path, is_chatml):
        print(f"[{name}]검사 중...")
        
        leak_count = 0
        total_scannet = 0
        
        # 파일 로드
        items = []
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r') as f:
                items = [json.loads(line) for line in f]
        else:
            with open(file_path, 'r') as f:
                items = json.load(f)
                
        # 검사
        for item in items:
            path = extract_path_from_chatml(item) if is_chatml else extract_path_from_raw(item)
            scene_id = get_scene_id(path)
            
            if scene_id:
                total_scannet += 1
                if scene_id in train_scenes:
                    leak_count += 1
        
        if total_scannet == 0:
            print(f"   ⚠️ ScanNet 데이터가 없습니다.")
            return

        leak_rate = (leak_count / total_scannet) * 100
        print(f"   - 전체 ScanNet 문제 수: {total_scannet}")
        print(f"   - 유출된 문제 수 (Train에서 본 장소): {leak_count}")
        print(f"   - 누수율 (Cheating Rate): {leak_rate:.2f}%")
        
        print("-" * 50)

    # 3. Val, Test 검사 실행
    analyze_split("Validation", VAL_FILE, is_chatml=True)
    analyze_split("Test (Hidden)", TEST_FILE, is_chatml=False)

if __name__ == "__main__":
    check_leakage()