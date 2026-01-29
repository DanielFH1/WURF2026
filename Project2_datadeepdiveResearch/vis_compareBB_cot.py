import matplotlib.pyplot as plt
import numpy as np

# ========================================================
# 📊 실험 3개 비교 데이터
# ========================================================

# 1. 카테고리
categories = [
    "Camera\nObj View",       
    "Camera\nRel Dir",       
    "Person\nObj View",       
    "Person\nRel Dir",       
    "Person\nScene Sim",      
    "TOTAL\nACCURACY"         
]

# 2. 데이터 입력 (Hardcoded)
# (1) Baseline (Vanilla)
baseline_scores = [70.70, 59.40, 89.10, 59.30, 67.00, 67.69]

# (2) Visual Prompt (Set-of-Mark)
# - 특징: Grounding(위치 찾기) 필요한 Task에서 강세, 전체 맥락에서 약세
visual_scores =   [72.83, 58.89, 86.96, 61.73, 56.88, 65.88]

# (3) CoT (Chain-of-Thought)
# - 특징: 시각 정보 없이 말만 길어져서 전체적으로 성능 하락 (Hallucination)
cot_scores =      [60.87, 55.00, 77.17, 48.15, 45.87, 56.86]

# ========================================================

def plot_triple_benchmark():
    # 그래프 설정
    x = np.arange(len(categories))
    width = 0.25  # 막대 폭 조절

    # 1. 텍스트 리포트
    print("\n" + "="*80)
    print(f"📊 Triple Comparison: Baseline vs Visual Prompt vs CoT")
    print("="*80)
    print(f"{'Category':<25} | {'Base':<7} | {'Visual':<7} | {'CoT':<7} | {'Best Method'}")
    print("-" * 80)
    
    for i, cat in enumerate(categories):
        cat_name = cat.replace('\n', ' ')
        b = baseline_scores[i]
        v = visual_scores[i]
        c = cot_scores[i]
        
        # 최고 성능 찾기
        scores = {'Base': b, 'Visual': v, 'CoT': c}
        best_method = max(scores, key=scores.get)
        best_score = scores[best_method]
        
        # Best Method 표시는 색깔 대신 텍스트로
        print(f"{cat_name:<25} | {b:<7.2f} | {v:<7.2f} | {c:<7.2f} | {best_method} ({best_score:.2f}%)")
        
    print("="*80)

    # 2. 그래프 그리기
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 색상 테마
    c_base = '#A9A9A9'   # 회색 (Baseline)
    c_vis = '#d62728'    # 빨강 (Visual Prompt - 강렬함)
    c_cot = '#1f77b4'    # 파랑 (CoT - 차분함/논리)

    # 막대 생성 (위치 조정: x-width, x, x+width)
    rects1 = ax.bar(x - width, baseline_scores, width, label='Baseline', color=c_base, alpha=0.7)
    rects2 = ax.bar(x, visual_scores, width, label='Visual Prompt', color=c_vis, alpha=0.9)
    rects3 = ax.bar(x + width, cot_scores, width, label='Chain-of-Thought', color=c_cot, alpha=0.7)

    # 그래프 꾸미기
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Impact of Spatial Strategies on VLM Performance', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 값 표시 함수
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Insight 박스 추가
    textstr = '\n'.join((
        r'$\bf{Analysis}$:',
        r'- Base: Best overall stability',
        r'- Visual: Strong in "Relative Direction"',
        r'- CoT: Degraded due to hallucination',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    output_path = "triple_comparison.svg"
    plt.savefig(output_path)
    print(f"\n🖼️  Graph saved to: {output_path}")

if __name__ == "__main__":
    plot_triple_benchmark()