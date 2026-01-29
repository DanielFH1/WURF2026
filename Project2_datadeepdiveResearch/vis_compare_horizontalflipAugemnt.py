import matplotlib.pyplot as plt
import numpy as np

# ========================================================
# 📊 데이터 입력
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

# 2. Baseline 점수 (Vanilla)
# Total = (65+107+82+48+73)/554 = 375/554 = 67.69%
baseline_scores = [
    70.70,  # Cam - Obj View
    59.40,  # Cam - Rel Dir
    89.10,  # Per - Obj View
    59.30,  # Per - Rel Dir
    67.00,  # Per - Scene Sim
    67.69   # Total
]

# 3. Augmented 점수 (Ours)
# Total = 372/554 = 67.15%
augmented_scores = [
    68.48,  # Cam - Obj View
    63.33,  # Cam - Rel Dir
    88.04,  # Per - Obj View
    61.73,  # Per - Rel Dir
    58.72,  # Per - Scene Sim
    67.15   # Total
]

# ========================================================

def plot_benchmark():
    # 그래프 설정
    x = np.arange(len(categories))
    width = 0.35

    # 1. 텍스트 리포트 출력
    print("\n" + "="*65)
    print(f"📊 Benchmark Comparison (with Total Accuracy)")
    print("="*65)
    print(f"{'Category':<30} | {'Base(%)':<8} | {'Aug(%)':<8} | {'Diff'}")
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
    
    # 막대 생성
    # Total Accuracy는 색상을 조금 진하게 해서 구별
    colors_base = ['#A9A9A9'] * 5 + ['#696969'] # 마지막만 진한 회색
    colors_aug = ['#1f77b4'] * 5 + ['#00008B']  # 마지막만 진한 파랑

    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color=colors_base, alpha=0.8)
    rects2 = ax.bar(x + width/2, augmented_scores, width, label='Augmented (Ours)', color=colors_aug)

    # 그래프 꾸미기
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Benchmark Performance Comparison (Baseline vs Augmented)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105) # 위 공간 확보
    
    # 범례 (Total 색상 구분을 위해 커스텀 핸들 대신 기본값 사용하되, 대표색으로 표시)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A9A9A9', label='Baseline'),
        Patch(facecolor='#1f77b4', label='Augmented (Ours)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 막대 위에 값 및 차이 표시 함수
    def autolabel(rects, scores, is_augmented=False):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            score = scores[idx]
            
            if is_augmented:
                # Augmented: 점수 + 증감폭(괄호)
                diff = score - baseline_scores[idx]
                diff_text = f"({diff:+.1f})"
                
                # 증감폭 색상: 상승(파랑/초록), 하락(빨강)
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

    autolabel(rects1, baseline_scores, is_augmented=False)
    autolabel(rects2, augmented_scores, is_augmented=True)

    # Total 부분 강조 박스 (선택 사항)
    # ax.axvline(x=4.5, color='black', linestyle=':', alpha=0.5) # Total 구분선

    plt.tight_layout()
    output_path = "benchmark_total_comparison."
    plt.savefig(output_path)
    print(f"\n🖼️  Graph saved to: {output_path}")

if __name__ == "__main__":
    plot_benchmark()