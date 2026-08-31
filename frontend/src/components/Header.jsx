import React from 'react';

export default function Header() {
  return (
    <header className="header">
      <div className="header-logo">
        <div className="header-logo-icon">S</div>
        <div>
          <div className="header-title">SARC</div>
          <div className="header-subtitle">Semantically-Aware Regional Compression</div>
        </div>
      </div>
      <div className="header-status">
        <span className="status-dot"></span>
        <span>AI Engine Ready</span>
      </div>
    </header>
  );
}
