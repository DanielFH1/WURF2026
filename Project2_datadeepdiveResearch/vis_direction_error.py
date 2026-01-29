import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ================= 데이터 설정 =================
# 로그에서 나온 수치를 그대로 입력하되, 라벨(이름)을 '정확한 의미'로 수정했습니다.
data = {
    'Error Type': [
        'Horizontal Flip\n(Left $\leftrightarrow$ Right)', 
        'Orthogonal Error\n(e.g., Front $\leftrightarrow$ Left)', 
        'Depth Error\n(Front $\leftrightarrow$ Back)',  # Vertical -> Depth 로 수정
        'Adjacent Error\n(e.g., Front $\leftrightarrow$ Front-Left)'
    ],
    'Count': [54, 41, 32, 3],
    'Percentage': [41.54, 31.54, 24.62, 2.31]
}

# 결과 저장 경로
RESULT_DIR = "data_divedive_results/direction_analysis_v4"
os.makedirs(RESULT_DIR, exist_ok=True)

def draw_chart():
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    # 그래프 스타일
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 7))
    
    # 막대 그래프 그리기 (1등인 Horizontal만 붉은색 강조)
    colors = ['#FF6B6B' if i == 0 else '#4A90E2' for i in range(len(df))]
    bars = plt.bar(df['Error Type'], df['Count'], color=colors, edgecolor='black', alpha=0.8, width=0.6)
    
    # 제목 및 축 설정
    plt.title('Distribution of Directional Errors (Revised Analysis)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Number of Errors', fontsize=12)
    plt.xlabel('Error Category', fontsize=12)
    
    # 막대 위에 수치 표시 (개수 + 퍼센트)
    for bar, count, pct in zip(bars, df['Count'], df['Percentage']):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 1,
            f'{count}\n({pct}%)',
            ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='black'
        )
    
    # Y축 여유 공간
    plt.ylim(0, max(df['Count']) * 1.15)
    plt.grid(axis='x') # 세로 격자 제거
    
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(RESULT_DIR, "directional_error_analysis_plot.svg")
    plt.savefig(save_path, dpi=300)
    print(f"✨ 그래프 저장 완료: {save_path}")

if __name__ == "__main__":
    draw_chart()