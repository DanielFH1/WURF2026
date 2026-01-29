import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. 데이터 입력 (User Results)
# ==========================================
categories = [
    'Overall\nAverage',
    '[Person]\nScene Sim.',
    '[Person]\nRel. Direction',
    '[Camera]\nObj. Orient.',
    '[Person]\nObj. Orient.',
    '[Camera]\nRel. Direction'
]

# Vanilla (Baseline) - 아까 구한 수치
vanilla_scores = [34.85, 35.22, 28.46, 30.06, 38.03, 39.08]

# MVSM (Fine-tuned) - 방금 구한 수치
mvsm_scores = [63.75, 59.12, 61.79, 67.05, 79.58, 56.70]

# 성능 향상폭 계산
gains = [m - v for m, v in zip(mvsm_scores, vanilla_scores)]

# ==========================================
# 2. 시각화 (Grouped Bar Chart)
# ==========================================
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))

# 막대 그리기
rects1 = ax.bar(x - width/2, vanilla_scores, width, label='Vanilla Qwen2.5-3B (Baseline)', color='#B0B0B0', alpha=0.6)
rects2 = ax.bar(x + width/2, mvsm_scores, width, label='MVSM Qwen2.5-3B (Ours)', color='#FF5733')

# 꾸미기
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Final Reproduction Result: Baseline vs Fine-tuned (MVSM)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.legend(fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# 논문 최고 성능(82%) 점선 표시 (참고용)
ax.axhline(y=82, color='green', linestyle=':', linewidth=2, label='Paper SOTA (Ref: 82%)')
ax.text(0, 83, 'Paper Best Reported (Maybe 72B/GPT-4)', color='green', fontsize=10)

# 막대 위에 점수 및 향상폭 표시
def autolabel(rects, is_mvsm=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        label = f'{height:.1f}%'
        
        # MVSM 막대 위에는 향상폭(+%)도 같이 표시
        if is_mvsm:
            gain = gains[i]
            ax.annotate(f'{height:.1f}%\n(+{gain:.1f}%)',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#D32F2F')
        else:
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='black')

autolabel(rects1)
autolabel(rects2, is_mvsm=True)

# -------------------------------------------------------
# 하단 테이블 추가
# -------------------------------------------------------
cell_text = [
    [f"{v:.2f}%" for v in vanilla_scores],
    [f"{m:.2f}%" for m in mvsm_scores],
    [f"+{g:.2f}%" for g in gains]
]
rows = ['Vanilla (Baseline)', 'MVSM (Ours)', 'Improvement']

table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  colLabels=categories,
                  loc='bottom',
                  bbox=[0.0, -0.3, 1.0, 0.2],
                  cellLoc='center')
table.set_fontsize(11)
table.scale(1, 1.5)

plt.subplots_adjust(left=0.1, bottom=0.25)
plt.savefig("final_reproduction_success.png", dpi=300)
print("✨ 시각화 완료! 'final_reproduction_success.png'를 확인하세요.")