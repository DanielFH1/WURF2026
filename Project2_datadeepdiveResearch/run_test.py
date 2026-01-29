import os
import subprocess
import multiprocessing
import time
import pandas as pd
import matplotlib.pyplot as plt

# ================= 설정 (Configuration) =================
# 사용할 GPU 번호
GPU_VANILLA = "2"
GPU_MVSM = "3"

# 모델 및 데이터 경로
DATASET_PATH = "data_train/test_hidden.json"
IMAGE_FOLDER = "ViewSpatial-Bench"

# 모델 경로
MODEL_VANILLA = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_MVSM = "./checkpoints/mvsm_merged"

# 결과 파일 예상 경로 (evaluate.py의 저장 규칙에 따름)
CSV_VANILLA = "result/Qwen2.5-VL-3B-Instruct/result_Qwen2.5-VL-3B-Instruct.csv"
CSV_MVSM = "result/mvsm_merged/result_mvsm_merged.csv"
# =======================================================

def run_evaluation(gpu_id, model_path, log_file):
    """
    지정된 GPU에서 evaluate.py를 실행하고 로그를 남기는 함수
    """
    print(f"🚀 [GPU {gpu_id}] Start evaluating: {model_path}")
    
    # 환경변수 설정 (해당 GPU만 보이게)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    cmd = [
        "python", "evaluate.py",
        "--model_path", model_path,
        "--dataset_path", DATASET_PATH,
        "--image_folder", IMAGE_FOLDER
    ]
    
    # 로그 파일 열고 서브프로세스 실행
    with open(log_file, "w") as f:
        # stdout과 stderr를 모두 로그 파일로 보냄
        process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        process.wait() # 끝날 때까지 대기
        
    print(f"✅ [GPU {gpu_id}] Finished: {model_path} (Log: {log_file})")

def visualize_results():
    """
    CSV 결과를 읽어서 비교 그래프를 그리는 함수
    """
    print("\n📊 Generating comparison plot...")
    
    # 데이터 로드 함수
    def get_acc(csv_path, label):
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing result file: {csv_path}")
            return 0.0
        try:
            df = pd.read_csv(csv_path)
            # 문자열 'True'/'False' 처리
            if df['IsCorrect'].dtype == 'object':
                df['IsCorrect'] = df['IsCorrect'].map({'True': True, 'False': False, 'TRUE': True, 'FALSE': False})
            acc = df['IsCorrect'].mean() * 100
            return acc
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return 0.0

    acc_vanilla = get_acc(CSV_VANILLA, "Vanilla")
    acc_mvsm = get_acc(CSV_MVSM, "MVSM")

    print(f"🔹 Vanilla Accuracy: {acc_vanilla:.2f}%")
    print(f"🔸 MVSM Accuracy:    {acc_mvsm:.2f}%")

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    models = ['Vanilla (Qwen2.5)', 'MVSM (Fine-tuned)']
    accs = [acc_vanilla, acc_mvsm]
    colors = ['gray', '#FF5733']

    bars = plt.bar(models, accs, color=colors, width=0.5)

    # 막대 위에 점수 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.2f}%',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.title(f"Performance Comparison: Baseline vs Fine-tuned\n(Gain: +{acc_mvsm - acc_vanilla:.2f}%)", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    output_img = "final_result_graph.png"
    plt.savefig(output_img, dpi=300)
    print(f"✨ Graph saved to: {output_img}")

if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 병렬 처리 프로세스 생성
    p1 = multiprocessing.Process(
        target=run_evaluation, 
        args=(GPU_VANILLA, MODEL_VANILLA, "eval_vanilla.log")
    )
    p2 = multiprocessing.Process(
        target=run_evaluation, 
        args=(GPU_MVSM, MODEL_MVSM, "eval_mvsm.log")
    )

    # 2. 실행 시작
    p1.start()
    p2.start()

    # 3. 끝날 때까지 대기 (Join)
    p1.join()
    p2.join()
    
    print("\n✅ All evaluations finished!")
    
    # 4. 결과 시각화
    visualize_results()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total elapsed time: {elapsed/60:.2f} minutes")