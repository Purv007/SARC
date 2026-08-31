import React, { useState } from 'react';
import ComparisonSlider from './ComparisonSlider';

export default function ResultsPanel({ results }) {
  const [activeTab, setActiveTab] = useState('comparison');

  if (!results) return null;

  const paths = results.output_paths;

  const tabs = [
    { id: 'comparison', label: '🔄 Compare' },
    { id: 'heatmap', label: '🌡️ Heatmap' },
    { id: 'detections', label: '🎯 Detections' },
    { id: 'quality', label: '📊 Quality Map' },
    { id: 'sarc', label: '✨ SARC Output' },
    { id: 'uniform', label: '📦 Uniform' },
  ];

  return (
    <div className="card animate-in">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="card-header-icon">🖼️</div>
          <span className="card-title">Results</span>
        </div>
        <a 
          href={paths.sarc_compressed} 
          download={`sarc_${results.image_info.filename}`}
          className="btn btn-primary"
          style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem', textDecoration: 'none' }}
        >
          ⬇️ Download
        </a>
      </div>
      <div className="card-body">
        <div className="tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div style={{ marginTop: '1rem' }}>
          {activeTab === 'comparison' && (
            <ComparisonSlider
              leftImage={paths.uniform_compressed}
              rightImage={paths.sarc_compressed}
              leftLabel="Uniform Compression"
              rightLabel="SARC Adaptive"
              isVideo={results.image_info.is_video}
            />
          )}

          {activeTab === 'heatmap' && (
            <div className="image-viewer">
              <img src={paths.importance_heatmap} alt="Importance Heatmap" />
              <div style={{ padding: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                🔴 Red = High importance (preserved) &nbsp; 🔵 Blue = Low importance (compressed more)
              </div>
            </div>
          )}

          {activeTab === 'detections' && (
            <>
              <div className="image-viewer">
                <img src={paths.detection_overlay} alt="Detection Overlay" />
              </div>
              <div style={{ marginTop: '1rem' }}>
                <DetectionList results={results} />
              </div>
            </>
          )}

          {activeTab === 'quality' && (
            <div className="image-viewer">
              <img src={paths.quality_map} alt="Quality Map" />
              <div style={{ padding: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                🔴 Red = High JPEG quality &nbsp; 🔵 Blue = Low JPEG quality (more compression)
              </div>
            </div>
          )}

          {activeTab === 'sarc' && (
            <div className="image-viewer">
              {results.image_info.is_video ? (
                <video src={paths.sarc_compressed} controls loop autoPlay style={{ width: '100%', borderRadius: '8px' }} />
              ) : (
                <img src={paths.sarc_compressed} alt="SARC Compressed" />
              )}
              <div style={{ padding: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                SARC adaptive compression — {results.metrics.sarc.file_size_kb} KB
                ({results.metrics.sarc.bandwidth_savings_percent}% smaller than original)
              </div>
            </div>
          )}

          {activeTab === 'uniform' && (
            <div className="image-viewer">
              {results.image_info.is_video ? (
                <video src={paths.uniform_compressed} controls loop autoPlay style={{ width: '100%', borderRadius: '8px' }} />
              ) : (
                <img src={paths.uniform_compressed} alt="Uniform Compressed" />
              )}
              <div style={{ padding: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Uniform JPEG compression (Q={results.config.uniform_quality}) — {results.metrics.uniform.file_size_kb} KB
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DetectionList({ results }) {
  const objects = results.detections?.objects || [];
  const textRegions = results.detections?.text_regions || [];

  if (objects.length === 0 && textRegions.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        No detections found in this image.
      </div>
    );
  }

  return (
    <div className="detection-list">
      {objects.map((obj, i) => (
        <div key={`obj-${i}`} className="detection-item">
          <span className={`detection-badge ${obj.importance >= 0.7 ? 'high' : obj.importance >= 0.4 ? 'medium' : 'low'}`}>
            {(obj.importance * 100).toFixed(0)}%
          </span>
          <span className="detection-name">🎯 {obj.class_name}</span>
          <span className="detection-confidence">{(obj.confidence * 100).toFixed(0)}% conf</span>
        </div>
      ))}
      {textRegions.map((text, i) => (
        <div key={`text-${i}`} className="detection-item">
          <span className="detection-badge high">100%</span>
          <span className="detection-name">📝 {text.text.substring(0, 30)}{text.text.length > 30 ? '...' : ''}</span>
          <span className="detection-confidence">{(text.confidence * 100).toFixed(0)}% conf</span>
        </div>
      ))}
    </div>
  );
}
