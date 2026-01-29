#!/bin/bash

# ================= 설정 =================
# 감시할 GPU 번호 (여기서는 2번과 3번)
GPU_IDS="2,3"

# 실행할 파이썬 명령어
CMD="python train_soft_distill.py"

# 학습 시작에 필요한 최소 메모리 (단위: MiB)
# Teacher(GPU2)는 15GB, Student(GPU3)는 24GB 정도 필요하다고 가정하고 여유 있게 잡음
REQ_MEM_GPU2=20000  # 20GB
REQ_MEM_GPU3=30000  # 30GB
# =======================================

echo "🔍 GPU $GPU_IDS 메모리 모니터링 시작..."
echo "   - GPU 2 필요: ${REQ_MEM_GPU2} MiB"
echo "   - GPU 3 필요: ${REQ_MEM_GPU3} MiB"

while true; do
    # 1. 현재 남은 메모리 조회 (nvidia-smi)
    # GPU 2번의 남은 용량
    FREE_MEM_2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)
    # GPU 3번의 남은 용량
    FREE_MEM_3=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 3)

    echo "[$(date '+%H:%M:%S')] 현재 여유공간 - GPU2: ${FREE_MEM_2} MiB / GPU3: ${FREE_MEM_3} MiB"

    # 2. 조건 확인 (둘 다 충분한가?)
    if [ "$FREE_MEM_2" -ge "$REQ_MEM_GPU2" ] && [ "$FREE_MEM_3" -ge "$REQ_MEM_GPU3" ]; then
        echo "✅ 공간 확보됨! 학습을 시작합니다. 🚀"
        echo "==========================================="
        
        # GPU 지정하고 실행
        CUDA_VISIBLE_DEVICES=2,3 $CMD
        
        # 학습이 끝나면 스크립트 종료 (에러로 꺼진거면 다시 대기하려면 이 부분 수정 가능)
        break
    else
        # 3. 충분하지 않으면 대기
        echo "⏳ 메모리 부족. 다른 프로세스가 끝나길 기다립니다... (60초 대기)"
        sleep 60
    fi
done