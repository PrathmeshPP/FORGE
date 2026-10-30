import React from 'react';
import {
  UploadCloud,
  Layers,
  Box,
  Cpu,
  ShieldCheck,
  Activity,
} from 'lucide-react';
import type { HealthResponse } from '../types';

export type ActiveTab = 'upload' | 'preflight' | 'status' | 'viewer' | 'library' | 'diagnostics';

interface HeaderProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  health: HealthResponse | null;
  activeJobId: string | null;
  hasReadyResult: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onSelectTab,
  health,
  activeJobId,
  hasReadyResult,
}) => {
  return (
    <header className="app-header">
      <div className="header-left">
        <div className="brand-badge">
          <Box size={18} className="brand-icon" />
          <div className="brand-text">
            <span className="brand-title">F.O.R.G.E.</span>
            <span className="brand-subtitle">Indoor 3D Reconstruction</span>
          </div>
        </div>
        <div className="boundary-badge" title="Workstream Boundary">
          <span>Website v1.0</span>
          <span className="divider">&middot;</span>
          <span className="subtext">Private Tailnet</span>
        </div>
      </div>

      <nav className="header-nav">
        <button
          className={`nav-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => onSelectTab('upload')}
          title="Upload Indoor Video"
        >
          <UploadCloud size={15} />
          <span>Upload</span>
        </button>

        {activeJobId && (
          <button
            className={`nav-btn ${activeTab === 'status' ? 'active' : ''}`}
            onClick={() => onSelectTab('status')}
            title="View Active Reconstruction Status"
          >
            <Activity size={15} className="pulse-icon" />
            <span>Processing</span>
          </button>
        )}

        <button
          className={`nav-btn ${activeTab === 'viewer' ? 'active' : ''}`}
          onClick={() => onSelectTab('viewer')}
          title="Interactive 3D Inspection Viewer"
        >
          <Box size={15} />
          <span>3D Viewer</span>
          {hasReadyResult && <span className="ready-dot" title="Model Loaded" />}
        </button>

        <button
          className={`nav-btn ${activeTab === 'library' ? 'active' : ''}`}
          onClick={() => onSelectTab('library')}
          title="Job Queue and History"
        >
          <Layers size={15} />
          <span>Jobs</span>
        </button>

        <button
          className={`nav-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
          onClick={() => onSelectTab('diagnostics')}
          title="Host & Security Diagnostics"
        >
          <Cpu size={15} />
          <span>Host Status</span>
        </button>
      </nav>

      <div className="header-right">
        <div className="host-status-indicator" title="Local Workstation Loopback Status">
          <span className={`status-dot ${health?.worker === 'busy' ? 'busy' : 'ready'}`} />
          <div className="host-meta">
            <span className="host-label">
              {health?.worker === 'busy' ? 'GPU Reconstructing' : 'Host Online (127.0.0.1)'}
            </span>
            <span className="host-mode">
              <ShieldCheck size={11} style={{ display: 'inline', marginRight: 3, verticalAlign: -1 }} />
              Mock Worker Active
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};

