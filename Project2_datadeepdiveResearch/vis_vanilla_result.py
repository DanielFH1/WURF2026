import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. 데이터 정의 (논문 수치 vs 내 수치)
# ==========================================
categories = [
    'Overall\nAverage',
    '[Camera]\nRel. Direction',
    '[Camera]\nObj. Orientation',
    '[Person]\nRel. Direction',
    '[Person]\nObj. Orientation',
    '[Person]\nScene Sim.'
]

# 논문 공식 수치 (Qwen2.5-VL-3B Backbone)
scores_paper = [35.85, 43.43, 33.33, 28.62, 39.16, 28.51]

# 사용자 재구현 수치
scores_user = [34.85, 39.08, 30.06, 28.46, 38.03, 35.22]

# ==========================================
# 2. 시각화 (막대 그래프 + 테이블)
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(categories))
width = 0.35

# 막대 그리기
rects1 = ax.bar(x - width/2, scores_paper, width, label='Paper Reported (Official)', color='#A9A9A9', alpha=0.8)
rects2 = ax.bar(x + width/2, scores_user, width, label='My Reproduction (Ours)', color='#4A90E2')

# 축 설정
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Reproduction Analysis: Paper vs User (Qwen2.5-VL-3B)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 55)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 막대 위에 점수 표시 함수
def autolabel(rects, is_user=False):
    for rect in rects:
        height = rect.get_height()
        # 글자 색상 및 스타일
        color = 'blue' if is_user else 'black'
        weight = 'bold' if is_user else 'normal'
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color=color, fontweight=weight)

autolabel(rects1)
autolabel(rects2, is_user=True)

# -------------------------------------------------------
# 하단에 데이터 테이블 추가 (논문 Table 형식 재현)
# -------------------------------------------------------
# 테이블 데이터 준비
cell_text = []
rows = ['Paper Reported', 'My Reproduction', 'Difference']
diffs = [f"{u - p:+.2f}%" for p, u in zip(scores_paper, scores_user)]

cell_text.append([f"{s:.2f}" for s in scores_paper])
cell_text.append([f"{s:.2f}" for s in scores_user])
cell_text.append(diffs)

# 테이블 그리기
table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  colLabels=categories,
                  loc='bottom',
                  bbox=[0.0, -0.35, 1.0, 0.25], # 위치 조절 [x, y, width, height]
                  cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)

# 레이아웃 조정 (테이블 잘림 방지)
plt.subplots_adjust(left=0.1, bottom=0.25)

# ==========================================
# 3. 저장
# ==========================================
save_path = "comparison_paper_vs_user.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✨ 비교 분석 완료! 결과가 '{save_path}'에 저장되었습니다.")