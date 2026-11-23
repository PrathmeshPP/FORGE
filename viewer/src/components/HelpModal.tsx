import React from 'react';
import { X, Navigation, Keyboard } from 'lucide-react';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Navigation size={18} color="var(--accent-cyan)" />
            <span style={{ fontWeight: 700, fontSize: '0.95rem', color: '#fff' }}>
              F.O.R.G.E Explorer Navigation & Controls
            </span>
          </div>
          <button className="btn-icon" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          {/* First-Person Walkthrough & Free-Roam Controls */}
          <div>
            <div style={{ fontWeight: 600, color: '#38bdf8', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Keyboard size={15} />
              <span>First-Person Room Walkthrough & Free-Roam (Exact Controls)</span>
            </div>
            <div className="shortcut-row">
              <span>Move Forward</span>
              <span className="kbd">W</span>
            </div>
            <div className="shortcut-row">
              <span>Move Backward</span>
              <span className="kbd">S</span>
            </div>
            <div className="shortcut-row">
              <span>Move Left (Strafe)</span>
              <span className="kbd">A</span>
            </div>
            <div className="shortcut-row">
              <span>Move Right (Strafe)</span>
              <span className="kbd">D</span>
            </div>
            <div className="shortcut-row">
              <span>Move Vertically Upward into the Air</span>
              <span className="kbd" style={{ color: '#4ade80' }}>U</span>
            </div>
            <div className="shortcut-row">
              <span>Move Vertically Downward</span>
              <span className="kbd" style={{ color: '#facc15' }}>P</span>
            </div>
            <div className="shortcut-row">
              <span>Sprint / Move Faster</span>
              <span className="kbd">Shift</span>
            </div>
            <div className="shortcut-row">
              <span>First-Person Mouse Look (Yaw & Pitch)</span>
              <span className="kbd">Mouse (Click Viewport)</span>
            </div>
            <div className="shortcut-row">
              <span>Release Mouse / Pause / Exit Walkthrough</span>
              <span className="kbd">ESC / G</span>
            </div>
          </div>

          {/* Mouse & Orbit Controls */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontWeight: 600, color: '#fff', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Navigation size={15} />
              <span>Orbit Inspection Mode (When Walkthrough is Off)</span>
            </div>
            <div className="shortcut-row">
              <span>Orbit / Rotate</span>
              <span className="kbd">Left Click + Drag</span>
            </div>
            <div className="shortcut-row">
              <span>Pan Viewport</span>
              <span className="kbd">Right Click + Drag</span>
            </div>
            <div className="shortcut-row">
              <span>Zoom In / Out</span>
              <span className="kbd">Scroll Wheel</span>
            </div>
          </div>

          {/* Inspection View Shortcuts */}
          <div style={{ marginTop: '1rem' }}>
            <div style={{ fontWeight: 600, color: '#fff', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Keyboard size={15} />
              <span>Inspection View Hotkeys</span>
            </div>
            <div className="shortcut-row">
              <span>Camera Presets</span>
              <span className="kbd">1 (Top) · 2 (Front) · 3 (Side) · 4 (Iso)</span>
            </div>
            <div className="shortcut-row">
              <span>Reset Camera</span>
              <span className="kbd">R</span>
            </div>
            <div className="shortcut-row">
              <span>Toggle Wireframe</span>
              <span className="kbd">W</span>
            </div>
            <div className="shortcut-row">
              <span>Toggle Grid</span>
              <span className="kbd">G</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
