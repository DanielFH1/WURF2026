import matplotlib.pyplot as plt
import numpy as np

# ========================================================
# 📊 데이터 입력 (Set-of-Mark 실험 결과 반영)
# ========================================================

# 1. 카테고리 (Total 포함)
categories = [
    "Camera\nObj View",       # Camera - Object View
    "Camera\nRel Dir",        # Camera - Relative Direction
    "Person\nObj View",       # Person - Object View
    "Person\nRel Dir",        # Person - Relative Direction
    "Person\nScene Sim",      # Person - Scene Simulation
    "TOTAL\nACCURACY"         # 전체 정확도
]

# 2. Baseline 점수 (Vanilla - 기존 값 유지)
baseline_scores = [
    70.70,  # Cam - Obj View
    59.40,  # Cam - Rel Dir
    89.10,  # Per - Obj View
    59.30,  # Per - Rel Dir
    67.00,  # Per - Scene Sim
    67.69   # Total
]

# 3. Ours 점수 (Visual Prompt - 방금 나온 결과값)
# Total = 365/554 = 65.88%
augmented_scores = [
    72.83,  # Cam - Obj View (67/92)
    58.89,  # Cam - Rel Dir (106/180)
    86.96,  # Per - Obj View (80/92) -> 소폭 하락했으나 여전히 높음
    61.73,  # Per - Rel Dir (50/81) -> 상승!
    56.88,  # Per - Scene Sim (62/109) -> 하락
    65.88   # Total
]

# ========================================================

def plot_benchmark():
    # 그래프 설정
    x = np.arange(len(categories))
    width = 0.35

    # 1. 텍스트 리포트 출력
    print("\n" + "="*65)
    print(f"📊 Benchmark Comparison: Baseline vs Visual Prompt (Set-of-Mark)")
    print("="*65)
    print(f"{'Category':<30} | {'Base(%)':<8} | {'Ours(%)':<8} | {'Diff'}")
    print("-" * 65)
    
    for i, cat in enumerate(categories):
        cat_name = cat.replace('\n', ' ')
        base = baseline_scores[i]
        aug = augmented_scores[i]
        diff = aug - base
        
        # Total 행은 구분선 추가
        if i == len(categories) - 1:
            print("-" * 65)
            
        print(f"{cat_name:<30} | {base:<8.2f} | {aug:<8.2f} | {diff:+.2f}%")
        
    print("="*65)

    # 2. 그래프 그리기
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # 막대 색상 설정
    # Baseline: 회색
    colors_base = ['#A9A9A9'] * 5 + ['#696969'] # 마지막만 진한 회색
    # Ours: 빨간색 계열 (Visual Prompt 강조)
    colors_aug = ['#ff7f0e'] * 5 + ['#d62728']  # 마지막만 진한 빨강 (오렌지 -> 레드)

    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color=colors_base, alpha=0.8)
    rects2 = ax.bar(x + width/2, augmented_scores, width, label='Visual Prompt (Ours)', color=colors_aug)

    # 그래프 꾸미기
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Performance Comparison: Baseline vs Visual Prompting')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105) # 위 공간 확보
    
    # 범례 커스텀
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A9A9A9', label='Baseline'),
        Patch(facecolor='#ff7f0e', label='Visual Prompt (Ours)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 막대 위에 값 및 차이 표시 함수
    def autolabel(rects, scores, is_ours=False):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            score = scores[idx]
            
            if is_ours:
                # Ours: 점수 + 증감폭(괄호)
                diff = score - baseline_scores[idx]
                diff_text = f"({diff:+.1f})"
                
                # 증감폭 색상: 상승(파랑), 하락(빨강)
                text_color = 'blue' if diff >= 0 else 'red'
                
                # 메인 텍스트 (점수)
                ax.text(rect.get_x() + rect.get_width()/2, height + 1,
                        f"{score:.1f}%",
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # 서브 텍스트 (증감폭)
                ax.text(rect.get_x() + rect.get_width()/2, height + 4,
                        diff_text,
                        ha='center', va='bottom', fontsize=9, color=text_color, fontweight='bold')
            else:
                # Baseline: 점수만 표시
                ax.text(rect.get_x() + rect.get_width()/2, height + 1,
                        f"{score:.1f}%",
                        ha='center', va='bottom', color='gray', fontsize=9)

    autolabel(rects1, baseline_scores, is_ours=False)
    autolabel(rects2, augmented_scores, is_ours=True)

    plt.tight_layout()
    output_path = "visual_prompt_comparison.svg"
    plt.savefig(output_path)
    print(f"\n🖼️  Graph saved to: {output_path}")

if __name__ == "__main__":
    plot_benchmark()