import matplotlib.pyplot as plt
import numpy as np

# ================= 데이터 입력 =================
# 방금 얻으신 실험 결과 (Leakage 모델)
datasets = ['Validation Set', 'Test Set']
scores = [73.20, 71.68] 

# 비교군 (논문 수치)
sota_score = 82.09  # 논문 MVSM (SOTA)
random_score = 26.33 # Random Baseline

# ================= 시각화 =================
fig, ax = plt.subplots(figsize=(9, 7))

# 1. 막대 그래프 그리기
# 색상: Validation(파랑), Test(주황) - 둘 다 진하게 표시
colors = ['#2b5c8a', '#d95f02']
bars = ax.bar(datasets, scores, color=colors, width=0.5, edgecolor='black', alpha=0.9, zorder=3)

# 2. 기준선 (Reference Lines)
# SOTA 라인
ax.axhline(sota_score, color='grey', linestyle='--', linewidth=2, zorder=2)
ax.text(1.3, sota_score + 1, f'Paper SOTA ({sota_score}%)', color='grey', fontweight='bold', va='bottom')

# Random 라인
ax.axhline(random_score, color='red', linestyle=':', linewidth=2, zorder=2)
ax.text(1.3, random_score + 1, f'Random Guess ({random_score}%)', color='red', fontweight='bold', va='bottom')

# 3. 수치 텍스트 표시
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 5,
            f'{height:.2f}%',
            ha='center', va='top', color='white', fontweight='bold', fontsize=16)

# 4. Leakage 강조 주석 (Gap 표시)
gap = abs(scores[0] - scores[1])
mid_x = 0.5 # 두 막대 사이
mid_y = max(scores) + 5

# 두 막대를 잇는 선과 텍스트
ax.annotate(f'Gap: Only {gap:.2f}%p\n(Evidence of Leakage)',
            xy=(0.5, max(scores)), xytext=(0.5, max(scores) + 12),
            arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=0.5', lw=1.5, color='black'),
            ha='center', fontsize=12, fontweight='bold', color='#c0392b')

# 5. 꾸미기
ax.set_ylim(0, 100)
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Result Analysis: Leaked Model Performance', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 저장
plt.tight_layout()
plt.savefig('vis_leakage_check.png', dpi=300)
print("✨ 그래프 저장 완료: vis_leakage_check.png")