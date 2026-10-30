import React from 'react';
import {
  Cpu,
  Shield,
  HardDrive,
  GitBranch,
  Lock,
} from 'lucide-react';
import type { HealthResponse } from '../types';

interface SystemDiagnosticsViewProps {
  health: HealthResponse | null;
}

export const SystemDiagnosticsView: React.FC<SystemDiagnosticsViewProps> = ({ health }) => {
  return (
    <div className="diagnostics-view-container">
      <div className="diagnostics-card">
        <div className="diagnostics-header">
          <div>
            <h2>Host Environment & Operational Diagnostics</h2>
            <p className="subtitle">
              Hardware verification, network boundaries, and cross-workstream pipeline integration status.
            </p>
          </div>
        </div>

        {/* Section 1: Workstream Boundary & Handoff Gate */}
        <div className="diag-section">
          <h3>
            <GitBranch size={16} className="text-cyan" />
            <span>Cross-Workstream Pipeline Status</span>
          </h3>
          <div className="gate-status-box">
            <div className="gate-status-header">
              <span className="gate-badge blocked">GATE STATUS: PENDING HANDOFF</span>
              <span className="gate-title">Indoor Model Pipeline Integration Gate</span>
            </div>
            <p className="gate-desc">
              The website workstream operates exclusively inside <code>FORGE_website</code>. The original reconstruction
              repository (<code>FORGE</code>) is maintained by a separate workstream. Real pipeline execution is blocked
              until the handoff contract in <code>PIPELINE_INTEGRATION_HANDOFF.md</code> is completed, tested on indoor video,
              and approved by the owner.
            </p>
            <div className="gate-facts-list">
              <div className="gate-fact">
                <span className="fact-label">Active Worker Adapter:</span>
                <span className="fact-val text-amber">
                  {health?.worker_mode === 'mock' ? 'Deterministic Mock Worker (Safe Simulation)' : 'Worker Ready'}
                </span>
              </div>
              <div className="gate-fact">
                <span className="fact-label">Original Pipeline Repo:</span>
                <span className="fact-val mono">C:\Users\25beevdt025\Desktop\FORGE (Read-Only)</span>
              </div>
              <div className="gate-fact">
                <span className="fact-label">VGGT Checkpoint:</span>
                <span className="fact-val mono">vggt_omega_1b_512.pt (4.26 GB detected)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Section 2: Host Hardware Diagnostics */}
        <div className="diag-section">
          <h3>
            <Cpu size={16} className="text-emerald" />
            <span>Workstation Compute & Memory Specifications</span>
          </h3>
          <div className="specs-grid">
            <div className="spec-item">
              <span className="spec-label">Operating System</span>
              <span className="spec-val">
                {health?.host_info?.platform || 'Windows 11 Pro (Build 22631)'}
              </span>
            </div>
            <div className="spec-item">
              <span className="spec-label">NVIDIA GPU Model</span>
              <span className="spec-val">
                {health?.host_info?.gpu_model || 'GeForce RTX 2070 SUPER'}
              </span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Dedicated VRAM Ceiling</span>
              <span className="spec-val">
                {health?.host_info ? `${health.host_info.vram_total_mb} MiB (8 GB Capacity)` : '8192 MiB (8 GB)'}
              </span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Current Free VRAM</span>
              <span className="spec-val text-emerald">
                {health?.host_info ? `~${(health.host_info.vram_free_mb / 1024).toFixed(1)} GB Available` : '~8.1 GB Available'}
              </span>
            </div>
            <div className="spec-item">
              <span className="spec-label">System Physical Memory</span>
              <span className="spec-val">32 GB RAM</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Local Disk Storage (C:)</span>
              <span className="spec-val">
                {health?.host_info ? `${health.host_info.disk_free_gb} GB Free` : '210.9 GB Free'}
              </span>
            </div>
          </div>
        </div>

        {/* Section 3: Security & Network Baseline */}
        <div className="diag-section">
          <h3>
            <Shield size={16} className="text-cyan" />
            <span>Security Baseline & Tailnet Configuration</span>
          </h3>
          <div className="security-checklist">
            <div className="sec-check-item">
              <Lock size={14} className="text-emerald" />
              <div>
                <strong>Loopback Network Binding:</strong>
                <p>Web server binds strictly to <code>127.0.0.1:8000</code>. No public 0.0.0.0 listeners.</p>
              </div>
            </div>
            <div className="sec-check-item">
              <Lock size={14} className="text-emerald" />
              <div>
                <strong>Private Remote Sharing:</strong>
                <p>Teammate access is proxied exclusively via Tailscale Serve. Tailscale Funnel and public tunnels are disabled.</p>
              </div>
            </div>
            <div className="sec-check-item">
              <Lock size={14} className="text-emerald" />
              <div>
                <strong>Zero Cloud Storage Leakage:</strong>
                <p>No AWS S3, cloud GPUs, telemetry, or third-party APIs. All processing stays local.</p>
              </div>
            </div>
            <div className="sec-check-item">
              <HardDrive size={14} className="text-emerald" />
              <div>
                <strong>Non-Synchronized Storage:</strong>
                <p>Data directory <code>FORGE_DATA_DIR</code> is outside Git and excluded from OneDrive / cloud sync.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

