import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import ResultsPanel from './components/ResultsPanel';
import MetricsDashboard from './components/MetricsDashboard';
import { compressImage } from './utils/api';
import HistoryPanel from './components/HistoryPanel';

const DEFAULT_CONFIG = {
  w_obj: 0.35,
  w_text: 0.30,
  w_edge: 0.15,
  w_spatial: 0.10,
  w_context: 0.10,
  context_factor: 1.0,
  q_min: 20,
  q_max: 95,
  uniform_quality: 75,
  boundary_sigma: 15,
};

export default function App() {
  const [file, setFile] = useState(null);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [results, setResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [processingStage, setProcessingStage] = useState('');

  const handleUpload = useCallback((uploadedFile) => {
    setFile(uploadedFile);
    setResults(null);
    setError(null);
  }, []);

  const handleConfigChange = useCallback((key, value) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleSelectHistory = useCallback((historyResult) => {
    setResults(historyResult);
    setFile(null); // Clear current upload
    setError(null);
  }, []);

  const handleProcess = useCallback(async () => {
    if (!file) return;

    setIsProcessing(true);
    setError(null);
    setProcessingStage('Uploading image...');

    try {
      setProcessingStage('Running AI detection & compression...');
      const result = await compressImage(file, config);
      setResults(result);
      setProcessingStage('');
    } catch (err) {
      setError(err.message || 'Processing failed');
      setProcessingStage('');
    } finally {
      setIsProcessing(false);
    }
  }, [file, config]);

  return (
    <>
      <Header />

      {/* Processing overlay */}
      {isProcessing && (
        <div className="processing-overlay">
          <div className="processing-content">
            <div className="processing-spinner" />
            <div className="processing-text">Processing with SARC</div>
            <div className="processing-subtext">{processingStage}</div>
            <div className="processing-subtext" style={{ marginTop: '0.5rem', opacity: 0.5 }}>
              This may take 10-30 seconds for high-resolution images
            </div>
          </div>
        </div>
      )}

      <div className="app-container">
        <div className="main-grid">
          {/* Left sidebar — Upload & Config */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <ImageUploader
              onUpload={handleUpload}
              onConfigChange={handleConfigChange}
              config={config}
              isProcessing={isProcessing}
            />

            {file && (
              <button
                className="btn btn-primary btn-full"
                onClick={handleProcess}
                disabled={isProcessing}
                style={{ padding: '1rem' }}
              >
                {isProcessing ? '⏳ Processing...' : '🚀 Run SARC Compression'}
              </button>
            )}

            {error && (
              <div style={{
                padding: '1rem',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: 'var(--accent-danger)',
                fontSize: '0.85rem',
              }}>
                ⚠️ {error}
              </div>
            )}

            {/* Quick stats from config */}
            {!results && file && (
              <div className="card">
                <div className="card-header">
                  <div className="card-header-icon">⚙️</div>
                  <span className="card-title">SARC Configuration</span>
                </div>
                <div className="card-body" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                    <div>Object: <strong>{config.w_obj}</strong></div>
                    <div>Text: <strong>{config.w_text}</strong></div>
                    <div>Edge: <strong>{config.w_edge}</strong></div>
                    <div>Spatial: <strong>{config.w_spatial}</strong></div>
                    <div>Quality: <strong>{config.q_min}-{config.q_max}</strong></div>
                    <div>Uniform Q: <strong>{config.uniform_quality}</strong></div>
                  </div>
                </div>
              </div>
            )}

            {/* History Panel */}
            <HistoryPanel onSelectExperiment={handleSelectHistory} />
          </div>

          {/* Right main — Results */}
          <div className="results-container">
            {!results && !file && (
              <div className="card">
                <div className="card-body">
                  <div className="empty-state">
                    <div className="empty-state-icon">🧠</div>
                    <div className="empty-state-title">SARC Compression Engine</div>
                    <div className="empty-state-text">
                      Upload an image to start. SARC will detect important regions using AI,
                      compute importance scores, and apply adaptive compression that preserves
                      what matters most.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!results && file && (
              <div className="card">
                <div className="card-body">
                  <div className="empty-state">
                    <div className="empty-state-icon">🚀</div>
                    <div className="empty-state-title">Ready to Process</div>
                    <div className="empty-state-text">
                      Click "Run SARC Compression" to analyze the image with AI detection,
                      compute the importance map, and generate adaptive compression.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {results && (
              <>
                <ResultsPanel results={results} />
                <MetricsDashboard results={results} />
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
