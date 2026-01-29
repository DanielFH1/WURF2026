import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 데이터 정의
# ==========================================

# (1) Paper SOTA (Table 2 Baseline)
data_paper = {
    "Overall": 82.09,
    "Cam_Rel": 83.59,
    "Cam_Obj": 87.65,
    "Per_Obj": 90.16,
    "Per_Rel": 71.14,
    "Per_Sim": 75.75
}

# (2) Before Augmentation (Epoch 10, Rank 64, Scene Split)
# User's previous result (71.30%)
data_no_aug = {
    "Overall": 71.30,
    "Cam_Rel": 62.22,
    "Cam_Obj": 79.35,
    "Per_Obj": 91.30,
    "Per_Rel": 61.73,
    "Per_Sim": 69.72
}

# (3) After Augmentation (Current Result)
# User's latest result (75.45%)
data_aug = {
    "Overall": 75.45,
    "Cam_Rel": 69.44,
    "Cam_Obj": 80.43,
    "Per_Obj": 91.30,
    "Per_Rel": 67.90,
    "Per_Sim": 73.39
}

# ==========================================
# 2. 데이터 정렬
# ==========================================
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

def extract_values(d):
    return [
        d["Overall"], 0, 
        d["Cam_Rel"], d["Cam_Obj"], 0, 
        d["Per_Obj"], d["Per_Rel"], d["Per_Sim"]
    ]

vals_paper = extract_values(data_paper)
vals_no_aug = extract_values(data_no_aug)
vals_aug = extract_values(data_aug)

# ==========================================
# 3. 시각화
# ==========================================
fig, ax = plt.subplots(figsize=(14, 9))

y = np.arange(len(labels))
height = 0.25

# 막대 그리기
# Paper (Grey)
rects1 = ax.barh(y - height, vals_paper, height, label='Paper SOTA', color='#9E9E9E', alpha=0.5)
# No Aug (Blue)
rects2 = ax.barh(y, vals_no_aug, height, label='No Augmentation (71.3%)', color='#4A90E2', alpha=0.8)
# With Aug (Red) - 주인공!
rects3 = ax.barh(y + height, vals_aug, height, label='With Augmentation (75.5%)', color='#D32F2F', alpha=1.0)

# 스타일 설정
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Data Augmentation: +4.15% Boost', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.set_xlim(0, 115)

# 값 표시 함수
def add_labels(rects, color, is_bold=False):
    for rect in rects:
        width = rect.get_width()
        if width > 0:
            fw = 'bold' if is_bold else 'normal'
            ax.text(width + 1, rect.get_y() + rect.get_height()/2, 
                    f'{width:.1f}%', 
                    va='center', ha='left', fontsize=9, fontweight=fw, color=color)

add_labels(rects1, 'black')
add_labels(rects2, 'black')
add_labels(rects3, '#D32F2F', is_bold=True) # 빨간색 강조

plt.tight_layout()
plt.savefig("vis_mix_epoch10r64.png")
print("✨ 최종 그래프 저장 완료: vis_mix_epoch10r64.png")