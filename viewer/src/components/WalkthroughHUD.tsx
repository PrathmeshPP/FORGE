import React from 'react';

interface WalkthroughHUDProps {
  isWalkthroughMode: boolean;
  onEnterWalkthrough: () => void;
  onExitWalkthrough: () => void;
  playerPos: [number, number, number];
  playerYaw: number;
  collisionStatus: 'CLEAR' | 'SLIDING' | 'BOUNDARY';
  fps: number;
  isPointerLocked: boolean;
  onRequestPointerLock: () => void;
}

export const WalkthroughHUD: React.FC<WalkthroughHUDProps> = ({
  isWalkthroughMode,
  onEnterWalkthrough,
  onExitWalkthrough,
  playerPos,
  playerYaw,
  collisionStatus,
  fps,
  isPointerLocked,
  onRequestPointerLock,
}) => {
  if (!isWalkthroughMode) {
    return (
      <div
        className="enter-space-container"
        style={{
          position: 'absolute',
          top: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 30,
        }}
      >
        <button
          onClick={onEnterWalkthrough}
          className="enter-space-btn"
          style={{
            background: 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
            border: '1px solid #38bdf8',
            borderRadius: '8px',
            color: '#ffffff',
            padding: '10px 24px',
            fontSize: '13px',
            fontWeight: 700,
            letterSpacing: '0.06em',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            boxShadow: '0 4px 20px rgba(2, 132, 199, 0.4), 0 0 10px rgba(56, 189, 248, 0.2)',
            transition: 'all 0.2s ease',
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path>
            <polyline points="10 17 15 12 10 7"></polyline>
            <line x1="15" y1="12" x2="3" y2="12"></line>
          </svg>
          ENTER INDOOR WALKTHROUGH
        </button>
      </div>
    );
  }

  const deg = Math.round(((-playerYaw * 180) / Math.PI) % 360);
  const normalizedDeg = deg < 0 ? deg + 360 : deg;

  const collisionColor =
    collisionStatus === 'CLEAR' ? '#4ade80' : collisionStatus === 'SLIDING' ? '#facc15' : '#f87171';

  return (
    <>
      {/* Reticle / Center Crosshair */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: 'rgba(255, 255, 255, 0.85)',
          border: '1px solid rgba(0, 0, 0, 0.5)',
          pointerEvents: 'none',
          zIndex: 20,
        }}
      />

      {/* Top Banner & Exit Button */}
      <div
        style={{
          position: 'absolute',
          top: '16px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 30,
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          background: 'rgba(15, 23, 42, 0.90)',
          border: '1px solid #334155',
          borderRadius: '8px',
          padding: '6px 16px',
          boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
          backdropFilter: 'blur(8px)',
          pointerEvents: 'auto',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', pointerEvents: 'none' }}>
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#22c55e',
              boxShadow: '0 0 8px #22c55e',
            }}
          />
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#f8fafc', letterSpacing: '0.04em' }}>
            IMMERSIVE INDOOR 3D WALKTHROUGH
          </span>
          <span style={{ fontSize: '11px', color: '#64748b' }}>|</span>
          <span style={{ fontSize: '11px', color: '#94a3b8' }}>HUMAN EYE-LEVEL</span>
        </div>

        <button
          className="exit-space-btn"
          onClick={() => {
            onExitWalkthrough();
          }}
          style={{
            background: 'rgba(239, 68, 68, 0.2)',
            border: '1px solid #ef4444',
            borderRadius: '6px',
            color: '#fca5a5',
            padding: '5px 12px',
            fontSize: '11px',
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            pointerEvents: 'auto',
          }}
          title="Exit Walkthrough (ESC / G)"
        >
          ✕ EXIT WALKTHROUGH [ESC]
        </button>
      </div>

      {/* Click-to-lock mouse overlay banner if pointer lock is lost */}
      {!isPointerLocked && (
        <div
          onClick={onRequestPointerLock}
          style={{
            position: 'absolute',
            top: '70px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 30,
            background: 'rgba(2, 132, 199, 0.92)',
            border: '1px solid #38bdf8',
            borderRadius: '6px',
            padding: '8px 20px',
            color: '#ffffff',
            fontSize: '12px',
            fontWeight: 700,
            cursor: 'pointer',
            boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
            pointerEvents: 'auto',
            letterSpacing: '0.03em',
          }}
        >
          CLICK VIEWPORT TO RESUME FIRST-PERSON MOUSE LOOK
        </div>
      )}

      {/* Bottom Visible Controls Bar */}
      <div
        style={{
          position: 'absolute',
          bottom: '24px',
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 25,
          background: 'rgba(15, 23, 42, 0.94)',
          border: '1px solid #38bdf8',
          borderRadius: '8px',
          padding: '10px 20px',
          display: 'flex',
          gap: '14px',
          fontSize: '12px',
          color: '#e2e8f0',
          boxShadow: '0 4px 20px rgba(0,0,0,0.6)',
          backdropFilter: 'blur(8px)',
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}
      >
        <span>
          <strong style={{ color: '#38bdf8', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #334155' }}>W</strong> Forward
        </span>
        <span style={{ color: '#475569' }}>•</span>
        <span>
          <strong style={{ color: '#38bdf8', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #334155' }}>S</strong> Backward
        </span>
        <span style={{ color: '#475569' }}>•</span>
        <span>
          <strong style={{ color: '#38bdf8', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #334155' }}>A</strong> Left
        </span>
        <span style={{ color: '#475569' }}>•</span>
        <span>
          <strong style={{ color: '#38bdf8', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #334155' }}>D</strong> Right
        </span>
        <span style={{ color: '#475569' }}>•</span>
        <span>
          <strong style={{ color: '#22c55e', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #166534' }}>U</strong> Up into Air
        </span>
        <span style={{ color: '#475569' }}>•</span>
        <span>
          <strong style={{ color: '#f59e0b', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #78350f' }}>P</strong> Down
        </span>
        <span style={{ color: '#475569' }}>•</span>
        <span>
          <strong style={{ color: '#a855f7', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #334155' }}>MOUSE</strong> Look Around
        </span>
        <span style={{ color: '#475569' }}>•</span>
        <span>
          <strong style={{ color: '#f87171', padding: '2px 5px', background: '#0f172a', borderRadius: '4px', border: '1px solid #7f1d1d' }}>ESC</strong> Exit
        </span>
      </div>

      {/* Telemetry Box (Top-Left) */}
      <div
        style={{
          position: 'absolute',
          top: '80px',
          left: '20px',
          zIndex: 25,
          background: 'rgba(15, 23, 42, 0.88)',
          border: '1px solid #1e293b',
          borderRadius: '6px',
          padding: '10px 14px',
          fontSize: '11px',
          color: '#94a3b8',
          fontFamily: 'monospace',
          lineHeight: 1.6,
          minWidth: '200px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #334155', paddingBottom: '4px', marginBottom: '4px' }}>
          <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>ROOM B 314 TELEMETRY</span>
          <span style={{ color: fps >= 50 ? '#4ade80' : fps >= 30 ? '#facc15' : '#f87171', fontWeight: 'bold' }}>
            {fps} FPS
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Position [X, Z]:</span>
          <span style={{ color: '#f8fafc' }}>
            [{playerPos[0].toFixed(2)}, {playerPos[2].toFixed(2)}]
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Vertical Altitude (Y):</span>
          <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>
            {playerPos[1].toFixed(3)} m
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Heading:</span>
          <span style={{ color: '#f8fafc' }}>{normalizedDeg}°</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Collision Status:</span>
          <span style={{ color: collisionColor, fontWeight: 'bold' }}>{collisionStatus}</span>
        </div>
      </div>
    </>
  );
};
