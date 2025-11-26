from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
import sys
import os
import tempfile

# UK_GeoAccent 프로젝트 루트를 Python path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 프로젝트 모듈 import
from preprocessing.preprocessing import AudioPreprocessor
# from models.baseline import Wav2Vec2Baseline  # 실제 모델 import (TODO)

app = Flask(__name__)
CORS(app)

# 설정
REGION_LABELS = ['Irish', 'Midland', 'Northern', 'Scottish', 'Southern', 'Welsh']
MODEL_PATH = os.path.join(project_root, 'models/best_model.pt')  # TODO: 실제 경로
USE_REAL_MODEL = False  # TODO: 모델 준비되면 True로 변경

# 전역 변수
model = None
preprocessor = None

def load_model():
    """모델 로드 함수"""
    global model, preprocessor
    
    if not USE_REAL_MODEL:
        print("⚠️  FALLBACK MODE: Using dummy data")
        return
    
    try:
        print("Loading model...")
        # TODO: 실제 모델 로드 코드
        # model = Wav2Vec2Baseline.from_pretrained(MODEL_PATH)
        # model.eval()
        
        preprocessor = AudioPreprocessor(sample_rate=16000)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("⚠️  Falling back to dummy mode")

# 서버 시작 시 모델 로드
load_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Accent Classifier API is running',
        'mode': 'model' if USE_REAL_MODEL and model is not None else 'fallback'
    })

@app.route('/api/classify', methods=['POST'])
def classify_accent():
    try:
        # 파일 확인
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # ========== 실제 모델 사용 ==========
        if USE_REAL_MODEL and model is not None:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                temp_path = tmp_file.name
            
            try:
                # 1. 오디오 로드 및 전처리
                waveform = preprocessor.load_audio(temp_path)
                waveform = preprocessor.normalize_audio(waveform)
                
                # 2. 모델 추론
                with torch.no_grad():
                    # TODO: 실제 모델 추론
                    # inputs = preprocessor.prepare_for_model(waveform)
                    # outputs = model(inputs)
                    # probabilities = torch.softmax(outputs['region_logits'], dim=-1)[0]
                    
                    # 임시 (위 주석 해제되면 삭제)
                    probabilities = torch.rand(len(REGION_LABELS))
                    probabilities = probabilities / probabilities.sum()
                
                # 3. 결과 생성
                predicted_idx = probabilities.argmax().item()
                predicted_accent = REGION_LABELS[predicted_idx]
                confidence = probabilities[predicted_idx].item()
                
                all_probabilities = {
                    label: round(prob.item(), 4)
                    for label, prob in zip(REGION_LABELS, probabilities)
                }
                
                mode = 'model'
                
            finally:
                # 임시 파일 삭제
                os.unlink(temp_path)
        
        # ========== Fallback 모드 ==========
        else:
            import time, random
            time.sleep(2)
            
            predicted_accent = random.choice(REGION_LABELS)
            probs = [random.random() for _ in REGION_LABELS]
            total = sum(probs)
            probs = [p/total for p in probs]
            
            predicted_idx = REGION_LABELS.index(predicted_accent)
            probs[predicted_idx] = max(probs) + 0.1
            
            total = sum(probs)
            probs = [p/total for p in probs]
            
            all_probabilities = {
                label: round(prob, 4) 
                for label, prob in zip(REGION_LABELS, probs)
            }
            
            confidence = all_probabilities[predicted_accent]
            mode = 'fallback'
        
        # 결과 반환
        result = {
            'accent': predicted_accent,
            'confidence': confidence,
            'allProbabilities': all_probabilities,
            'mode': mode
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print("Error during classification:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🎙️  Accent Classifier API Server")
    print("=" * 50)
    print("📍 Running on: http://localhost:5000")
    print(f"🔧 Mode: {'MODEL' if USE_REAL_MODEL else 'FALLBACK (dummy data)'}")
    print("=" * 50)
    app.run(debug=True, port=5000)