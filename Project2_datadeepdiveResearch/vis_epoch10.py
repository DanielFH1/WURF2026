import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 데이터 정의
# ==========================================

# (1) Paper SOTA (Table 2 Baseline)
# 순서: Overall, [Cam]Rel, [Cam]Obj, [Per]Obj, [Per]Rel, [Per]Sim
score_paper = {
    "Overall": 82.09,
    "Cam_Rel": 83.59,
    "Cam_Obj": 87.65,
    "Per_Obj": 90.16,
    "Per_Rel": 71.14,
    "Per_Sim": 75.75
}

# (2) LoRA Rank 64 (Previous Run - User Provided)
# User Input: [66.11, 75.00, 94.57, 70.37, 65.14] (순서: CamRel, CamObj, PerObj, PerRel, PerSim)
raw_lora64 = [66.11, 75.00, 94.57, 70.37, 65.14]
avg_lora64 = np.mean(raw_lora64)
score_lora64 = {
    "Overall": avg_lora64,
    "Cam_Rel": raw_lora64[0],
    "Cam_Obj": raw_lora64[1],
    "Per_Obj": raw_lora64[2],
    "Per_Rel": raw_lora64[3],
    "Per_Sim": raw_lora64[4]
}

# (3) My Model (Epoch 10 - Scene Split Corrected)
# User Input:
# Cam-Obj: 79.35 / Cam-Rel: 62.22
# Per-Obj: 91.30 / Per-Rel: 61.73 / Per-Sim: 69.72
# Overall: 71.30
score_epoch10 = {
    "Overall": 71.30,
    "Cam_Rel": 62.22,  # Relative Direction
    "Cam_Obj": 79.35,  # Object View Orientation
    "Per_Obj": 91.30,  # Object View Orientation
    "Per_Rel": 61.73,  # Relative Direction
    "Per_Sim": 69.72   # Scene Simulation
}

# ==========================================
# 2. 데이터 정렬 (시각화용 리스트 변환)
# ==========================================
# 그래프에 표시할 라벨 순서
labels = [
    'Overall Avg', 
    '---', 
    '[Cam] Rel.Dir', 
    '[Cam] Obj.Ori', 
    '---', 
    '[Per] Obj.Ori', 
    '[Per] Rel.Dir', 
    '[Per] Sce.Sim'
]

# 딕셔너리에서 순서대로 값 추출하는 헬퍼 함수
def extract_values(source_dict):
    return [
        source_dict["Overall"],
        0, # Separator
        source_dict["Cam_Rel"],
        source_dict["Cam_Obj"],
        0, # Separator
        source_dict["Per_Obj"],
        source_dict["Per_Rel"],
        source_dict["Per_Sim"]
    ]

data_paper = extract_values(score_paper)
data_lora64 = extract_values(score_lora64)
data_epoch10 = extract_values(score_epoch10)

# ==========================================
# 3. 시각화 그리기
# ==========================================
fig, ax = plt.subplots(figsize=(14, 9))

y = np.arange(len(labels))  # 라벨 위치
height = 0.25               # 막대 두께

# 막대 그리기 (위치 조정: y-height, y, y+height)
# 1. Paper (회색)
rects1 = ax.barh(y - height, data_paper, height, label='Paper SOTA (Baseline)', color='grey', alpha=0.4)
# 2. LoRA Rank 64 (파란색)
rects2 = ax.barh(y, data_lora64, height, label='My model (LoRA r64)', color='#4A90E2', alpha=0.8)
# 3. Epoch 10 (빨간색 - 주인공)
rects3 = ax.barh(y + height, data_epoch10, height, label='My model (Epoch 10)', color='#D32F2F')

# 축 및 라벨 설정
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()  # 위에서부터 Overall이 나오도록 반전
ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Paper vs LoRA (r64) vs Epoch10', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.set_xlim(0, 110) # 텍스트 공간 확보

# 수치 텍스트 추가 함수
def add_labels(rects, is_bold=False):
    for rect in rects:
        width = rect.get_width()
        if width > 0: # 0인 Separator는 건너뜀
            fw = 'bold' if is_bold else 'normal'
            ax.text(width + 1, rect.get_y() + rect.get_height()/2, 
                    f'{width:.2f}%', 
                    va='center', ha='left', fontsize=9, fontweight=fw, color='black')

add_labels(rects1, is_bold=False)
add_labels(rects2, is_bold=False)
add_labels(rects3, is_bold=True) # 현재 모델 강조

# ==========================================
# 4. 저장
# ==========================================
plt.tight_layout()
save_filename = "comparison_3models_final.png"
plt.savefig(save_filename, dpi=300)
print(f"✨ 그래프 저장 완료: {save_filename}")