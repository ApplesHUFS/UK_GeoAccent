from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import random

app = Flask(__name__)
CORS(app)

REGION_LABELS = ['Irish', 'Midland', 'Northern', 'Scottish', 'Southern', 'Welsh']

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Accent Classifier API is running'
    })

@app.route('/api/classify', methods=['POST'])
def classify_accent():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
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
        
        result = {
            'accent': predicted_accent,
            'confidence': confidence,
            'allProbabilities': all_probabilities,
            'mode': 'fallback'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Accent Classifier API Server")
    print("=" * 50)
    print("Running on: http://localhost:5000")
    print("Mode: FALLBACK (dummy data)")
    print("=" * 50)
    app.run(debug=True, port=5000)