import React from 'react';

export default function MetricsDashboard({ results }) {
  if (!results) return null;

  const sarc = results.metrics?.sarc;
  const uniform = results.metrics?.uniform;
  const comparison = results.comparison;
  const timing = results.timing;

  if (!sarc || !uniform) return null;

  return (
    <div className="card animate-in">
      <div className="card-header">
        <div className="card-header-icon">📊</div>
        <span className="card-title">Compression Metrics</span>
      </div>
      <div className="card-body">
        {/* Size comparison bar */}
        <SizeComparisonBar
          originalSize={results.image_info.original_size_bytes}
          uniformSize={uniform.file_size_bytes}
          sarcSize={sarc.file_size_bytes}
        />

        {/* Metrics grid */}
        <div className="metrics-grid" style={{ marginTop: '1.5rem' }}>
          <MetricCard
            label="SARC File Size"
            value={sarc.file_size_kb}
            unit="KB"
            type="neutral"
          />
          <MetricCard
            label="Uniform File Size"
            value={uniform.file_size_kb}
            unit="KB"
            type="neutral"
          />
          <MetricCard
            label="SARC vs Uniform"
            value={comparison.size_reduction_vs_uniform_percent > 0 ? '-' + comparison.size_reduction_vs_uniform_percent : '+' + Math.abs(comparison.size_reduction_vs_uniform_percent)}
            unit="%"
            type={comparison.size_reduction_vs_uniform_percent > 0 ? 'positive' : 'negative'}
          />
          <MetricCard
            label="SARC Bandwidth Saved"
            value={sarc.bandwidth_savings_percent}
            unit="%"
            type="positive"
          />
          <MetricCard
            label="SARC PSNR"
            value={sarc.psnr}
            unit="dB"
            type="neutral"
            comparison={comparison.psnr_difference !== null ? {
              value: comparison.psnr_difference,
              label: 'vs uniform',
              better: comparison.psnr_difference >= 0
            } : null}
          />
          <MetricCard
            label="SARC SSIM"
            value={sarc.ssim}
            unit=""
            type="neutral"
            comparison={comparison.ssim_difference !== null ? {
              value: comparison.ssim_difference,
              label: 'vs uniform',
              better: comparison.ssim_difference >= 0
            } : null}
          />
          {sarc.lpips !== null && (
            <MetricCard
              label="SARC LPIPS"
              value={sarc.lpips}
              unit=""
              type="neutral"
            />
          )}
          <MetricCard
            label="Compression Ratio"
            value={sarc.compression_ratio}
            unit="×"
            type="neutral"
          />
          <MetricCard
            label="Processing Time"
            value={timing.total_seconds}
            unit="sec"
            type="neutral"
          />
          <MetricCard
            label="Objects Detected"
            value={results.detections.total_objects}
            unit=""
            type="neutral"
          />
          <MetricCard
            label="Text Regions"
            value={results.detections.total_text_regions}
            unit=""
            type="neutral"
          />
          <MetricCard
            label="Avg Importance"
            value={results.importance_stats.mean}
            unit=""
            type="neutral"
          />
        </div>

        {/* Transmission time table */}
        {sarc.transmission_times && (
          <div style={{ marginTop: '1.5rem' }}>
            <h3 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              📡 Estimated Transmission Time Savings
            </h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                    <th style={thStyle}>Network</th>
                    <th style={thStyle}>Original</th>
                    <th style={thStyle}>SARC</th>
                    <th style={thStyle}>Saved</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(sarc.transmission_times).map(([key, times]) => (
                    <tr key={key} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={tdStyle}>{formatNetworkName(key)}</td>
                      <td style={tdStyle}>{times.original_seconds}s</td>
                      <td style={{ ...tdStyle, color: 'var(--accent-success)' }}>{times.compressed_seconds}s</td>
                      <td style={{ ...tdStyle, color: 'var(--accent-primary-light)' }}>{times.saved_seconds}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value, unit, type, comparison: comp }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${type}`}>
        {typeof value === 'number' ? (
          value < 1 && value > 0 ? value.toFixed(4) : value.toFixed(2)
        ) : value}
      </div>
      {unit && <div className="metric-unit">{unit}</div>}
      {comp && (
        <div className={`metric-comparison ${comp.better ? 'better' : 'worse'}`}>
          <span>{comp.better ? '▲' : '▼'}</span>
          <span>{comp.value > 0 ? '+' : ''}{typeof comp.value === 'number' ? comp.value.toFixed(3) : comp.value} {comp.label}</span>
        </div>
      )}
    </div>
  );
}

function SizeComparisonBar({ originalSize, uniformSize, sarcSize }) {
  const maxSize = originalSize;
  const uniformPct = (uniformSize / maxSize * 100).toFixed(1);
  const sarcPct = (sarcSize / maxSize * 100).toFixed(1);

  return (
    <div>
      <div className="comparison-bar-label">
        <span>File Size Comparison</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
          Original: {(originalSize / 1024).toFixed(0)} KB
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Uniform JPEG</div>
          <div className="comparison-bar">
            <div className="comparison-bar-segment uniform" style={{ width: `${uniformPct}%` }}>
              {uniformPct}%
            </div>
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>SARC Adaptive</div>
          <div className="comparison-bar">
            <div className="comparison-bar-segment sarc" style={{ width: `${sarcPct}%` }}>
              {sarcPct}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const thStyle = {
  textAlign: 'left',
  padding: '0.5rem',
  color: 'var(--text-muted)',
  fontWeight: 500,
  fontSize: '0.75rem',
};

const tdStyle = {
  padding: '0.5rem',
  fontFamily: 'var(--font-mono)',
  color: 'var(--text-secondary)',
};

function formatNetworkName(key) {
  const names = {
    'slow_3g_kbps': '3G Slow (400 Kbps)',
    'fast_3g_kbps': '3G Fast (1.5 Mbps)',
    '4g_kbps': '4G (10 Mbps)',
    'wifi_kbps': 'WiFi (50 Mbps)',
  };
  return names[key] || key;
}
