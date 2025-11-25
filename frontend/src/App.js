import React, { useState } from 'react';
import './App.css';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  // 파일 선택 핸들러
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('audio/')) {
      setAudioFile(file);
      setResult(null); // 새 파일 선택 시 이전 결과 초기화
    } else {
      alert('음성 파일을 선택해주세요.');
    }
  };

  // 분석 요청 핸들러
  // 분석 요청 핸들러
const handleAnalyze = async () => {
  if (!audioFile) {
    alert('먼저 음성 파일을 선택해주세요.');
    return;
  }

  setIsAnalyzing(true);

  try {
    // FormData로 파일 전송
    const formData = new FormData();
    formData.append('audio', audioFile);

    // Flask API 호출
    const response = await fetch('http://localhost:5000/api/classify', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('API 요청 실패');
    }

    const data = await response.json();
    setResult(data);
  } catch (error) {
    console.error('Error:', error);
    alert('분석 중 오류가 발생했습니다: ' + error.message);
  } finally {
    setIsAnalyzing(false);
  }
};

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎙️ UK Accent Classifier</h1>
        <p>영국 방언 분류기</p>
      </header>

      <main className="App-main">
        {/* 파일 업로드 섹션 */}
        <div className="upload-section">
          <label htmlFor="audio-upload" className="upload-label">
            음성 파일 선택
          </label>
          <input
            id="audio-upload"
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="file-input"
          />
          {audioFile && (
            <p className="file-name">선택된 파일: {audioFile.name}</p>
          )}
        </div>

        {/* 분석 버튼 */}
        <button
          onClick={handleAnalyze}
          disabled={!audioFile || isAnalyzing}
          className="analyze-button"
        >
          {isAnalyzing ? '분석 중...' : '분석하기'}
        </button>

        {/* 결과 표시 섹션 */}
        {result && (
          <div className="result-section">
            <h2>분석 결과</h2>
            <div className="main-result">
              <p className="accent-label">감지된 방언:</p>
              <p className="accent-value">{result.accent}</p>
              <p className="confidence">
                신뢰도: {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>

            <div className="all-probabilities">
              <h3>모든 지역 확률</h3>
              {Object.entries(result.allProbabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([accent, prob]) => (
                  <div key={accent} className="probability-bar">
                    <span className="accent-name">{accent}</span>
                    <div className="bar-container">
                      <div
                        className="bar-fill"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="probability-value">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;