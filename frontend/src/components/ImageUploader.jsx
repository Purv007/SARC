import React, { useState, useRef, useCallback } from 'react';

export default function ImageUploader({ onUpload, onConfigChange, config, isProcessing }) {
  const [preview, setPreview] = useState(null);
  const [fileInfo, setFileInfo] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file || !(file.type.startsWith('image/') || file.type.startsWith('video/'))) return;

    setFileInfo({
      name: file.name,
      size: (file.size / 1024).toFixed(1),
      type: file.type,
    });

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);

    onUpload(file);
  }, [onUpload]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-header-icon">📤</div>
        <span className="card-title">Upload Media</span>
      </div>
      <div className="card-body">
        {!preview ? (
          <div
            className={`upload-zone ${isDragging ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <div className="upload-icon">🖼️</div>
            <div className="upload-text">
              Drag & drop an image or video here or <strong style={{ color: 'var(--accent-primary-light)' }}>browse</strong>
            </div>
            <div className="upload-subtext">
              Supports JPEG, PNG, WebP, MP4, WebM
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/mp4,video/webm"
              onChange={(e) => handleFile(e.target.files[0])}
              style={{ display: 'none' }}
            />
          </div>
        ) : (
          <div className="upload-preview">
            {fileInfo?.type.startsWith('image/') ? (
              <img src={preview} alt="Upload preview" />
            ) : (
              <video src={preview} controls style={{ width: '100%' }} />
            )}
            <div className="upload-preview-info">
              <div><strong>{fileInfo?.name}</strong></div>
              <div>{fileInfo?.size} KB • {fileInfo?.type}</div>
            </div>
          </div>
        )}

        {preview && (
          <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem' }}>
            <button
              className="btn btn-secondary"
              style={{ flex: 1 }}
              onClick={() => { setPreview(null); setFileInfo(null); }}
            >
              Change Media
            </button>
            <button
              className="btn btn-secondary"
              style={{ flex: 'none' }}
              onClick={() => setShowConfig(!showConfig)}
            >
              ⚙️ {showConfig ? 'Hide' : 'Config'}
            </button>
          </div>
        )}

        {showConfig && (
          <div className="config-section">
            <ConfigSlider
              label="Object Weight"
              value={config.w_obj}
              onChange={(v) => onConfigChange('w_obj', v)}
            />
            <ConfigSlider
              label="Text Weight"
              value={config.w_text}
              onChange={(v) => onConfigChange('w_text', v)}
            />
            <ConfigSlider
              label="Edge Weight"
              value={config.w_edge}
              onChange={(v) => onConfigChange('w_edge', v)}
            />
            <ConfigSlider
              label="Spatial Weight"
              value={config.w_spatial}
              onChange={(v) => onConfigChange('w_spatial', v)}
            />
            <ConfigSlider
              label="Context Factor"
              value={config.context_factor}
              onChange={(v) => onConfigChange('context_factor', v)}
            />
            <ConfigSlider
              label="Min Quality"
              value={config.q_min}
              onChange={(v) => onConfigChange('q_min', v)}
              min={1}
              max={100}
              step={1}
            />
            <ConfigSlider
              label="Max Quality"
              value={config.q_max}
              onChange={(v) => onConfigChange('q_max', v)}
              min={1}
              max={100}
              step={1}
            />
            <ConfigSlider
              label="Uniform Quality"
              value={config.uniform_quality}
              onChange={(v) => onConfigChange('uniform_quality', v)}
              min={1}
              max={100}
              step={1}
            />
            <ConfigSlider
              label="Boundary Sigma"
              value={config.boundary_sigma}
              onChange={(v) => onConfigChange('boundary_sigma', v)}
              min={0}
              max={50}
              step={1}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function ConfigSlider({ label, value, onChange, min = 0, max = 1, step = 0.01 }) {
  return (
    <div className="config-group">
      <div className="config-label">
        <span>{label}</span>
        <span className="config-value">{typeof value === 'number' ? value.toFixed(step < 1 ? 2 : 0) : value}</span>
      </div>
      <input
        type="range"
        className="config-slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}
