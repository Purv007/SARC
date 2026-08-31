import React, { useState, useEffect } from 'react';
import { getExperiments, getExperiment, deleteExperiment } from '../utils/api';

export default function HistoryPanel({ onSelectExperiment }) {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const data = await getExperiments();
      setExperiments(data.experiments);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    if (window.confirm("Are you sure you want to delete this history item?")) {
      try {
        await deleteExperiment(id);
        fetchHistory();
      } catch (err) {
        alert("Failed to delete: " + err.message);
      }
    }
  };

  const handleSelect = async (id) => {
    try {
      const data = await getExperiment(id);
      
      // Adapt backend history response to match the /api/compress response structure
      // that ResultsPanel and MetricsDashboard expect.
      const adaptedResults = adaptHistoryToResults(data);
      onSelectExperiment(adaptedResults);
    } catch (err) {
      alert('Failed to load experiment details: ' + err.message);
    }
  };

  if (loading) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Loading history...</div>;
  }

  if (error) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--accent-danger)' }}>{error}</div>;
  }

  if (experiments.length === 0) {
    return (
      <div className="card animate-in">
        <div className="card-body">
          <div className="empty-state">
            <div className="empty-state-icon">📚</div>
            <div className="empty-state-title">No History Yet</div>
            <div className="empty-state-text">
              Run a compression task and it will be saved here automatically.
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card animate-in">
      <div className="card-header">
        <div className="card-header-icon">🕰️</div>
        <span className="card-title">Compression History</span>
      </div>
      <div className="card-body" style={{ padding: 0 }}>
        <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
          {experiments.map((exp) => {
            const sarcMetrics = exp.metrics?.find(m => m.method === 'sarc') || {};
            const savings = sarcMetrics.bandwidth_savings_percent || 0;
            
            return (
              <div 
                key={exp.id} 
                className="history-item"
                onClick={() => handleSelect(exp.id)}
                style={{
                  padding: '1rem 1.5rem',
                  borderBottom: '1px solid var(--border-color)',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  transition: 'background 0.2s ease'
                }}
                onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-glass-hover)'}
                onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.25rem' }}>
                    {exp.filename}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {new Date(exp.created_at).toLocaleString()} • {(exp.original_size_bytes / 1024).toFixed(1)} KB
                  </div>
                </div>
                <div style={{ textAlign: 'right', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div>
                    <div style={{ color: 'var(--accent-success)', fontWeight: 'bold', fontSize: '1.1rem' }}>
                      -{savings.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      Saved
                    </div>
                  </div>
                  <button 
                    onClick={(e) => handleDelete(e, exp.id)}
                    style={{
                      background: 'transparent', 
                      border: 'none', 
                      cursor: 'pointer', 
                      color: 'var(--accent-danger)', 
                      fontSize: '1.1rem',
                      opacity: 0.7,
                      padding: '0.25rem'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.opacity = '1'}
                    onMouseOut={(e) => e.currentTarget.style.opacity = '0.7'}
                    title="Delete history"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Helper function to adapt the database format back to the API format
function adaptHistoryToResults(dbExp) {
  const sarcMetrics = dbExp.metrics?.find(m => m.method === 'sarc') || {};
  const uniformMetrics = dbExp.metrics?.find(m => m.method === 'uniform') || {};
  
  const sarcResult = dbExp.compression_results?.find(c => c.method === 'sarc') || {};
  const uniformResult = dbExp.compression_results?.find(c => c.method === 'uniform') || {};
  
  const objects = dbExp.detections?.filter(d => d.detector_type === 'yolo').map(d => ({
    class_name: d.class_name,
    confidence: d.confidence,
    importance: d.importance_score,
    bbox: [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2]
  })) || [];
  
  const text_regions = dbExp.detections?.filter(d => d.detector_type === 'ocr').map(d => ({
    text: d.class_name, // OCR text was stored in class_name
    confidence: d.confidence,
    bbox: [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2]
  })) || [];
  // Derive quality_map path from sarc output path (same directory)
  const sarcDir = sarcResult.output_path ? sarcResult.output_path.replace(/\/[^/]+$/, '') : '';

  return {
    experiment_id: dbExp.id.toString(),
    image_info: {
      filename: dbExp.filename,
      width: dbExp.width,
      height: dbExp.height,
      original_size_bytes: dbExp.original_size_bytes,
      original_size_kb: dbExp.original_size_bytes / 1024,
    },
    detections: {
      objects,
      text_regions,
      total_objects: objects.length,
      total_text_regions: text_regions.length,
    },
    importance_stats: { mean: 0, min: 0, max: 1, std: 0 }, // Mock since we don't save this
    quality_stats: { mean_quality: 0, min_quality: 0, max_quality: 100 },
    metrics: {
      sarc: {
        psnr: sarcMetrics.psnr,
        ssim: sarcMetrics.ssim,
        lpips: sarcMetrics.lpips,
        file_size_bytes: sarcMetrics.file_size_bytes,
        file_size_kb: sarcMetrics.file_size_bytes / 1024,
        compression_ratio: sarcMetrics.compression_ratio,
        bandwidth_savings_percent: sarcMetrics.bandwidth_savings_percent,
        processing_time_seconds: sarcMetrics.processing_time_seconds,
        transmission_times: null,
      },
      uniform: {
        psnr: uniformMetrics.psnr,
        ssim: uniformMetrics.ssim,
        lpips: uniformMetrics.lpips,
        file_size_bytes: uniformMetrics.file_size_bytes,
        file_size_kb: uniformMetrics.file_size_bytes / 1024,
        compression_ratio: uniformMetrics.compression_ratio,
        bandwidth_savings_percent: uniformMetrics.bandwidth_savings_percent,
        processing_time_seconds: uniformMetrics.processing_time_seconds,
        transmission_times: null,
      }
    },
    comparison: {
      size_reduction_vs_uniform_percent: (1 - (sarcMetrics.file_size_bytes / uniformMetrics.file_size_bytes)) * 100,
      psnr_difference: sarcMetrics.psnr - uniformMetrics.psnr,
      ssim_difference: sarcMetrics.ssim - uniformMetrics.ssim,
    },
    timing: { total_seconds: sarcMetrics.processing_time_seconds || 0 },
    output_paths: {
      original: '',
      sarc_compressed: sarcResult.output_path || '',
      uniform_compressed: uniformResult.output_path || '',
      importance_heatmap: sarcResult.heatmap_path || '',
      quality_map: sarcDir ? `${sarcDir}/quality_map.jpg` : '',
      detection_overlay: sarcResult.overlay_path || '',
    },
    config: JSON.parse(dbExp.config_json || '{}')
  };
}
