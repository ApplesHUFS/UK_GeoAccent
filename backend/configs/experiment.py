"""
experiment.py
Runs all defined experiments (Baseline, Ablations, Final Model) sequentially.
"""
import subprocess
import os
import json
import sys

# 프로젝트의 메인 실행 파일 경로
MAIN_SCRIPT = 'main.py'
CONFIG_DIR = 'configs'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'results'

# 1. 실행할 실험 목록 정의
# key: 실험 이름, value: 사용할 JSON config 파일명
EXPERIMENTS = {
    "Final_Model": "final_model.json",
    "Baseline_Frozen": "baseline_frozen.json",
    "Ablation_No_Distance": "ablation_no_dist.json",
    "Ablation_No_Fusion": "ablation_no_geo.json",
}

def run_command(command, experiment_name):
    """주어진 쉘 명령어를 실행하고 결과를 출력합니다."""
    print(f"\n{'='*20} 🏃‍♂️ Starting {experiment_name} - {command[1]} {'='*20}")
    try:
        # subprocess.run을 사용하여 명령 실행
        # stdout=subprocess.PIPE, stderr=subprocess.PIPE를 사용하여 출력을 캡처할 수 있지만,
        # 여기서는 실시간 출력을 위해 그대로 둡니다.
        process = subprocess.run(
            command,
            check=True,  # 오류 발생 시 예외 발생
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"{'='*20} ✅ {experiment_name} - {command[1]} Complete {'='*20}")
    except subprocess.CalledProcessError as e:
        print(f"\n{'!'*20} ❌ ERROR during {experiment_name} - {command[1]} {'!'*20}")
        print(f"Command failed with error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n{'!'*20} ❌ ERROR: {MAIN_SCRIPT} not found. Ensure you are in the project root directory. {'!'*20}")
        sys.exit(1)


def get_best_checkpoint(experiment_name):
    """해당 실험의 최적 체크포인트 경로를 추론합니다."""
    # WandB/Logger가 checkpoint_dir/{experiment_name}/best.pt와 같은 구조로 저장한다고 가정
    best_path = os.path.join(CHECKPOINT_DIR, experiment_name, 'best.pt')
    
    # 실제로는 학습 로그를 분석하여 가장 좋은 체크포인트를 찾아야 하지만, 
    # 여기서는 Trainer가 'best.pt'를 저장한다고 가정합니다.
    if os.path.exists(best_path):
        return best_path
    
    # 🚨 주의: 실제 구현에서는 이 부분이 중요합니다. Trainer가 저장한 정확한 경로를 알아야 합니다.
    print(f"\n{'!'*10} WARNING: Could not find assumed best checkpoint at {best_path}. Please check trainer logic. {'!'*10}")
    # 일단 'last.pt'를 시도하거나, 혹은 스크립트 실행 중 수동으로 경로를 확인해야 합니다.
    # 안전을 위해 여기서 None을 반환하고 사용자에게 확인 요청
    return None 

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*80)
    print("       🔬 GeoAccent Classifier Full Experiment Suite Started 🔬")
    print("="*80)

    for exp_name, config_file in EXPERIMENTS.items():
        config_path = os.path.join(CONFIG_DIR, config_file)

        # 1. 학습 (TRAIN) 명령어 구성
        train_command = [
            sys.executable,  # 현재 활성화된 python 인터프리터 사용
            MAIN_SCRIPT,
            'train',
            '--config', config_path,
            '--wandb_run_name', exp_name # WandB 실행 이름으로 사용
            # 기타 필요한 인자 (예: --use_wandb)는 main.py에서 처리된다고 가정
        ]
        
        # 2. 학습 실행
        run_command(train_command, exp_name)

        # 3. 평가 (EVALUATE) 준비
        # 학습이 완료된 후, 최적 체크포인트 경로를 찾음
        best_checkpoint = get_best_checkpoint(exp_name)
        
        if best_checkpoint and os.path.exists(best_checkpoint):
            # 4. 평가 (EVALUATE) 명령어 구성
            eval_command = [
                sys.executable, 
                MAIN_SCRIPT,
                'evaluate',
                '--checkpoint', best_checkpoint,
                '--split', 'test', # 최종 성능 측정을 위해 test split 사용
                '--output_dir', os.path.join(RESULTS_DIR, f'{exp_name}_results')
            ]

            # 5. 평가 실행
            run_command(eval_command, exp_name)
        else:
            print(f"\n{'!'*20} Skipping EVALUATION for {exp_name} - Checkpoint not found. {'!'*20}")
            print(f"Please manually evaluate the best checkpoint for {exp_name}.")


if __name__ == "__main__":
    # experiment.py도 backend/ 디렉토리에서 실행해야 main.py를 찾을 수 있습니다.
    # 현재 디렉토리가 main.py가 있는 곳인지 확인
    if not os.path.exists(MAIN_SCRIPT):
        print(f"ERROR: {MAIN_SCRIPT} not found in the current directory.")
        print("Please execute this script from the project root (backend/) folder.")
        sys.exit(1)
        
    main()