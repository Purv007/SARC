import React, { useState, useRef, useCallback, useEffect } from 'react';

export default function ComparisonSlider({ leftImage, rightImage, leftLabel, rightLabel, isVideo }) {
  const containerRef = useRef(null);
  const [position, setPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const leftVideoRef = useRef(null);
  const rightVideoRef = useRef(null);

  const updatePosition = useCallback((clientX) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(5, Math.min(95, (x / rect.width) * 100));
    setPosition(pct);
  }, []);

  useEffect(() => {
    const handleMove = (e) => {
      if (!isDragging) return;
      e.preventDefault();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      updatePosition(clientX);
    };

    const handleUp = () => setIsDragging(false);

    if (isDragging) {
      window.addEventListener('mousemove', handleMove);
      window.addEventListener('mouseup', handleUp);
      window.addEventListener('touchmove', handleMove, { passive: false });
      window.addEventListener('touchend', handleUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
      window.removeEventListener('touchmove', handleMove);
      window.removeEventListener('touchend', handleUp);
    };
  }, [isDragging, updatePosition]);

  // Sync video playback
  useEffect(() => {
    if (isVideo && leftVideoRef.current && rightVideoRef.current) {
      const left = leftVideoRef.current;
      const right = rightVideoRef.current;
      
      const syncPlay = () => {
        if (Math.abs(left.currentTime - right.currentTime) > 0.1) {
          right.currentTime = left.currentTime;
        }
      };
      
      left.addEventListener('timeupdate', syncPlay);
      return () => left.removeEventListener('timeupdate', syncPlay);
    }
  }, [isVideo]);

  const handleStart = (e) => {
    setIsDragging(true);
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    updatePosition(clientX);
  };

  if (!leftImage || !rightImage) return null;

  return (
    <div
      ref={containerRef}
      className="comparison-container"
      onMouseDown={handleStart}
      onTouchStart={handleStart}
    >
      {/* Right media (full width background) */}
      {isVideo ? (
        <video 
          ref={rightVideoRef}
          src={rightImage} 
          style={{ width: '100%' }} 
          autoPlay loop muted playsInline 
        />
      ) : (
        <img src={rightImage} alt={rightLabel || 'Right'} style={{ width: '100%' }} />
      )}

      {/* Left media (clipped) */}
      <div
        className="comparison-overlay"
        style={{ width: `${position}%` }}
      >
        {isVideo ? (
          <video 
            ref={leftVideoRef}
            src={leftImage} 
            style={{ width: containerRef.current ? containerRef.current.offsetWidth : '100%' }} 
            autoPlay loop muted playsInline controls
          />
        ) : (
          <img
            src={leftImage}
            alt={leftLabel || 'Left'}
            style={{ width: containerRef.current ? containerRef.current.offsetWidth : '100%' }}
          />
        )}
      </div>

      {/* Divider line */}
      <div className="comparison-divider" style={{ left: `${position}%` }} />

      {/* Labels */}
      <span className="comparison-label comparison-label-left">{leftLabel || 'Before'}</span>
      <span className="comparison-label comparison-label-right">{rightLabel || 'After'}</span>
    </div>
  );
}
