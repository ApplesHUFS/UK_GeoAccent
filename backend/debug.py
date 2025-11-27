import os
import glob

# data 폴더의 기본 경로 (프로젝트 루트에서 실행한다고 가정)
base_path = 'data/english_dialects' 

splits = ['train', 'test', 'validation']

print("--- 데이터셋 분할 개수 ---")

for split in splits:
    # 각 분할 폴더의 경로
    folder_path = os.path.join(base_path, split)
    
    # 해당 폴더 내의 모든 파일 (재귀적 탐색 없이 1단계만 확인)
    # 실제 데이터 파일의 확장자(예: .wav, .mp3, .json 등)를 알고 있다면 아래 *.* 대신 명시하는 것이 좋습니다.
    # 만약 음성 파일(.wav)만 세고 싶다면: file_count = len(glob.glob(os.path.join(folder_path, '*.wav')))
    
    # 여기서는 폴더 내의 모든 항목을 세고, 디렉토리는 제외합니다.
    file_count = 0
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, item)):
                file_count += 1
    
    print(f"✅ {split} 분할: {file_count} 개")

print("----------------------")