import React from 'react';

export const ControlsFooter: React.FC = () => {
  return (
    <footer
      style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        width: '100%',
        padding: '12px 20px',
        background: 'linear-gradient(to top, rgba(9, 13, 22, 0.95) 0%, rgba(9, 13, 22, 0.8) 70%, transparent 100%)',
        zIndex: 20,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '6px',
        pointerEvents: 'none',
        userSelect: 'none',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          color: '#e2e8f0',
          fontSize: '13px',
          fontWeight: 500,
          fontFamily: 'system-ui, -apple-system, sans-serif',
          letterSpacing: '0.3px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <kbd style={kbdStyle}>W</kbd>
          <span style={labelStyle}>Forward</span>
        </div>

        <span style={dividerStyle}>•</span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <kbd style={kbdStyle}>S</kbd>
          <span style={labelStyle}>Backward</span>
        </div>

        <span style={dividerStyle}>•</span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <kbd style={kbdStyle}>A</kbd>
          <span style={labelStyle}>Left</span>
        </div>

        <span style={dividerStyle}>•</span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <kbd style={kbdStyle}>D</kbd>
          <span style={labelStyle}>Right</span>
        </div>

        <span style={dividerStyle}>•</span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <kbd style={kbdStyle}>Space</kbd>
          <span style={{ color: '#94a3b8' }}>/</span>
          <kbd style={kbdStyle}>Q</kbd>
          <span style={labelStyle}>Upward (In Air)</span>
        </div>

        <span style={dividerStyle}>•</span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <kbd style={kbdStyle}>Shift</kbd>
          <span style={{ color: '#94a3b8' }}>/</span>
          <kbd style={kbdStyle}>C</kbd>
          <span style={labelStyle}>Downward</span>
        </div>

        <span style={dividerStyle}>•</span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: '#38bdf8', fontWeight: 600 }}>🖱️ Click & Drag</span>
          <span style={labelStyle}>Look Around</span>
        </div>
      </div>

      <div
        style={{
          fontSize: '11px',
          color: '#64748b',
          fontFamily: 'system-ui, sans-serif',
        }}
      >
        Click anywhere on 3D view to lock cursor for first-person FPS look • Press <kbd style={{ ...kbdStyle, padding: '1px 4px', fontSize: '10px' }}>Esc</kbd> to unlock
      </div>
    </footer>
  );
};

const kbdStyle: React.CSSProperties = {
  background: '#1e293b',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  boxShadow: '0 2px 4px rgba(0,0,0,0.4)',
  borderRadius: '4px',
  padding: '2px 7px',
  color: '#38bdf8',
  fontWeight: 700,
  fontSize: '11px',
  fontFamily: 'monospace',
};

const labelStyle: React.CSSProperties = {
  color: '#cbd5e1',
};

const dividerStyle: React.CSSProperties = {
  color: 'rgba(255, 255, 255, 0.2)',
  fontSize: '14px',
};
