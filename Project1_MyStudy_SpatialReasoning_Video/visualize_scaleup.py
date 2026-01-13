import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

# 1. 데이터 로드
csv_path = "experiment_results_scaleup_gpu3.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Data Loaded: {len(df)} frames")
except FileNotFoundError:
    print("❌ Error: CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    # 테스트용 더미 데이터 생성 (파일 없을 때만 작동)
    df = pd.DataFrame({
        'time': np.linspace(0, 100, 1000),
        'base_pred': ['Left']*300 + ['Center']*20 + ['Left']*300 + ['Right']*380,
        'fixed_pred': ['Left']*320 + ['Left']*300 + ['Right']*380,
        'adapt_pred': ['Left']*320 + ['Left']*300 + ['Right']*380,
        'entropy': np.random.rand(1000),
        'used_alpha': np.random.rand(1000)
    })

# 2. TC-Score 계산 함수
def calculate_tc_score(preds):
    if len(preds) < 2: return 0.0
    # 문자열 비교
    changes = sum(1 for i in range(len(preds)-1) if preds[i] == preds[i+1])
    return changes / (len(preds) - 1)

tc_base = calculate_tc_score(df['base_pred'].tolist())
tc_fixed = calculate_tc_score(df['fixed_pred'].tolist())
tc_adapt = calculate_tc_score(df['adapt_pred'].tolist())

# ==========================================
# 🎨 Visualization: The "Mega-Figure"
# ==========================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2) # 3행 2열 레이아웃

# --- [A] Temporal Consistency Barcode (전체 흐름) ---
# Baseline vs Adaptive 비교 (가로로 긴 바코드 형태)
ax1 = fig.add_subplot(gs[0, :]) # 첫 줄 전체 사용

# 색상 매핑 (Left:빨강, Right:파랑, Center:초록, 기타:회색)
unique_labels = sorted(list(set(df['base_pred'].unique()) | set(df['adapt_pred'].unique())))
color_map = {'Left': '#FF5555', 'Right': '#5555FF', 'Center': '#55FF55', 'Front': 'orange', 'Back': 'purple'}
# 매핑되지 않은 단어는 회색 처리
colors = [color_map.get(lbl, 'lightgray') for lbl in unique_labels]
cmap = mcolors.ListedColormap(colors)

# 데이터 숫자로 변환
label_to_num = {lbl: i for i, lbl in enumerate(unique_labels)}
base_nums = df['base_pred'].map(label_to_num).values.reshape(1, -1)
adapt_nums = df['adapt_pred'].map(label_to_num).values.reshape(1, -1)

# 바코드 그리기
ax1.imshow(np.vstack([base_nums, adapt_nums]), aspect='auto', cmap=cmap, interpolation='nearest')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Baseline\n(Static)', 'Ours\n(Adaptive)'], fontsize=14, fontweight='bold')
ax1.set_xlabel("Time (Frame Index)", fontsize=12)
ax1.set_title(f"(A) Temporal Stability Visualization: Color Barcode (Total {len(df)} Frames)", fontsize=16, fontweight='bold')

# 범례 추가 (Custom Legend)
patches = [plt.Rectangle((0,0),1,1, color=color_map.get(l, 'lightgray')) for l in unique_labels if l in color_map]
ax1.legend(patches, [l for l in unique_labels if l in color_map], loc='upper right', ncol=len(unique_labels))


# --- [B] Adaptive Alpha Mechanism (작동 원리) ---
# 엔트로피가 높을 때 Alpha가 어떻게 변했나?
ax2 = fig.add_subplot(gs[1, 0])
sns.scatterplot(x=df['entropy'], y=df['used_alpha'], alpha=0.1, color='purple', ax=ax2)
ax2.set_xlabel("Prediction Entropy (Uncertainty)", fontsize=12)
ax2.set_ylabel("Adaptive Alpha value", fontsize=12)
ax2.set_title("(B) Mechanism: Higher Uncertainty → Stronger Memory", fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.5)
# 추세선 추가
sns.regplot(x=df['entropy'], y=df['used_alpha'], scatter=False, color='red', ax=ax2, line_kws={'linestyle':'--'})


# --- [C] Zoom-in View (특정 구간 확대) ---
# 엔트로피가 가장 높았던(가장 혼란스러웠던) 구간 200프레임 확대
max_entropy_idx = df['entropy'].idxmax()
start_zoom = max(0, max_entropy_idx - 100)
end_zoom = min(len(df), max_entropy_idx + 100)
zoom_df = df.iloc[start_zoom:end_zoom]

ax3 = fig.add_subplot(gs[1, 1])
# Alpha 값 변화 그래프
ax3.plot(zoom_df['time'], zoom_df['used_alpha'], color='purple', linewidth=2, label='Adaptive Alpha')
ax3.fill_between(zoom_df['time'], 0, zoom_df['used_alpha'], color='purple', alpha=0.1)
ax3.set_title(f"(C) Zoom-in: Alpha Response at High Uncertainty (t={zoom_df['time'].iloc[0]:.1f}s~)", fontsize=14)
ax3.set_ylabel("Alpha Value")
ax3.legend()
ax3.grid(True, linestyle=':')


# --- [D] Final Score Comparison (성적표) ---
ax4 = fig.add_subplot(gs[2, :]) # 마지막 줄 전체
scores = [tc_base, tc_fixed, tc_adapt]
methods = ['Baseline', 'Fixed Alpha (0.6)', 'Adaptive Alpha']
colors_bar = ['gray', 'royalblue', 'purple']

bars = ax4.barh(methods, scores, color=colors_bar, height=0.6)
ax4.set_xlim(0.8, 1.02) # 차이 잘 보이게 X축 조정 (데이터에 따라 조절 필요)
ax4.set_xlabel("Temporal Consistency Score (TC-Score)", fontsize=12)
ax4.set_title("(D) Quantitative Result: Robustness Comparison", fontsize=14)

# 막대 옆에 수치 표시
for bar, score in zip(bars, scores):
    ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
             f"{score:.4f}", va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("figure6_scaleup_result.png", dpi=300)
plt.show()

print("✅ Figure Saved: figure6_scaleup_result.png")