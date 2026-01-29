import matplotlib.pyplot as plt
import numpy as np

# 순서: [Cam]Rel, [Cam]Obj, [Per]Obj, [Per]Rel, [Per]Sim
user_scores = [66.11 , 75.00 , 94.57, 70.37, 65.14] 

# [Paper] 논문 SOTA (Table 2)
paper_scores = [83.59, 87.65, 90.16, 71.14, 75.75]
paper_avg = 82.09

# User 평균 계산
user_avg = np.mean(user_scores) if sum(user_scores) > 0 else 0

# ==========================================
# 시각화
# ==========================================
labels = ['Overall Avg', '---', '[Cam] Rel.Dir', '[Cam] Obj.Ori', '---', '[Per] Obj.Ori', '[Per] Rel.Dir', '[Per] Sce.Sim']
data_paper = [paper_avg, 0, paper_scores[0], paper_scores[1], 0, paper_scores[2], paper_scores[3], paper_scores[4]]
data_user = [user_avg, 0, user_scores[0], user_scores[1], 0, user_scores[2], user_scores[3], user_scores[4]]

fig, ax = plt.subplots(figsize=(12, 8))
y = np.arange(len(labels))
height = 0.35

ax.barh(y - height/2, data_paper, height, label='Paper (SOTA)', color='grey', alpha=0.5)
ax.barh(y + height/2, data_user, height, label='My Model (Rank 64)', color='red')

ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel('Accuracy (%)'); ax.set_title('Final Result: My Model vs Paper')
ax.legend(); ax.grid(axis='x', linestyle='--', alpha=0.3)

for i, v in enumerate(data_paper):
    if v > 0: 
        ax.text(v+1, i-height/2, f'{v:.1f}%', va='center', fontweight='normal', color='black', fontsize=9)

for i, v in enumerate(data_user):
    if v > 0: ax.text(v+1, i+height/2, f'{v:.1f}%', va='center', fontweight='bold', color='red')

plt.tight_layout()
plt.savefig("final_result_lora_rank64.png")
print("그래프 저장 완료: final_result_lora_rank64.png")