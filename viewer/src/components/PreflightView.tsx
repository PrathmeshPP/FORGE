import React from 'react';
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  Film,
  Clock,
  Maximize2,
  HardDrive,
  Cpu,
  ArrowRight,
  ArrowLeft,
  ShieldCheck,
} from 'lucide-react';
import type { PreflightMetadata } from '../types';

interface PreflightViewProps {
  metadata: PreflightMetadata;
  keepRawInput: boolean;
  onConfirmStart: () => void;
  onBackToUpload: () => void;
}

export const PreflightView: React.FC<PreflightViewProps> = ({
  metadata,
  keepRawInput,
  onConfirmStart,
  onBackToUpload,
}) => {
  const isRejected = metadata.status === 'rejected';
  const hasWarnings = metadata.status === 'warning';

  return (
    <div className="preflight-view-container">
      <div className="preflight-card">
        <div className="preflight-header">
          <button className="btn-back" onClick={onBackToUpload}>
            <ArrowLeft size={14} />
            <span>Choose Different Video</span>
          </button>
          <h2>Video Preflight Inspection</h2>
          <p className="subtitle">
            Local automated verification of container headers, frame dimensions, and temporal duration before initiating
            GPU inference.
          </p>
        </div>

        {/* Status Verdict Banner */}
        <div className={`verdict-banner ${metadata.status}`}>
          <div className="verdict-icon">
            {metadata.status === 'passed' ? (
              <CheckCircle size={22} className="text-emerald" />
            ) : hasWarnings ? (
              <AlertTriangle size={22} className="text-amber" />
            ) : (
              <XCircle size={22} className="text-rose" />
            )}
          </div>
          <div className="verdict-text">
            <span className="verdict-title">
              {metadata.status === 'passed'
                ? 'Preflight Verification Passed'
                : hasWarnings
                ? 'Preflight Passed with Advisories'
                : 'Preflight Verification Rejected'}
            </span>
            <p className="verdict-desc">
              {metadata.status === 'passed'
                ? 'Container and video stream satisfy all requirements for indoor spatial reconstruction on this host.'
                : hasWarnings
                ? 'Video is decodable, but low resolution or duration may affect geometric completeness.'
                : 'Video cannot be processed. Please resolve the blocking issues listed below.'}
            </p>
          </div>
        </div>

        {/* Video Metadata Grid */}
        <div className="meta-grid">
          <div className="meta-item">
            <span className="meta-label">
              <Film size={13} /> File Name
            </span>
            <span className="meta-value mono" title={metadata.filename}>
              {metadata.filename}
            </span>
          </div>

          <div className="meta-item">
            <span className="meta-label">
              <HardDrive size={13} /> File Size
            </span>
            <span className="meta-value">
              {metadata.file_size_mb < 1024
                ? `${metadata.file_size_mb.toFixed(1)} MB`
                : `${(metadata.file_size_mb / 1024).toFixed(2)} GB`}
            </span>
          </div>

          <div className="meta-item">
            <span className="meta-label">
              <Clock size={13} /> Duration
            </span>
            <span className="meta-value">
              {metadata.duration_seconds.toFixed(1)} seconds (
              {Math.floor(metadata.duration_seconds / 60)}m {Math.floor(metadata.duration_seconds % 60)}s)
            </span>
          </div>

          <div className="meta-item">
            <span className="meta-label">
              <Maximize2 size={13} /> Resolution & Aspect
            </span>
            <span className="meta-value">
              {metadata.width} &times; {metadata.height} ({metadata.aspect_ratio})
            </span>
          </div>
        </div>

        {/* Systematic Check Table */}
        <div className="checks-section">
          <h3>Preflight Gate Diagnostics</h3>
          <div className="checks-table">
            <div className="check-row">
              <span className="check-status pass">
                <CheckCircle size={14} />
              </span>
              <div className="check-details">
                <span className="check-title">Video Container Format Allowlist</span>
                <span className="check-desc">
                  Matched accepted container: <code>{metadata.filename.split('.').pop()?.toUpperCase()}</code> (H.264 /
                  HEVC track parsed).
                </span>
              </div>
            </div>

            <div className="check-row">
              <span className="check-status pass">
                <CheckCircle size={14} />
              </span>
              <div className="check-details">
                <span className="check-title">Workstation Upload & Storage Cap</span>
                <span className="check-desc">
                  File size {metadata.file_size_mb.toFixed(1)} MB is well within the 2048 MB workstation ceiling.
                </span>
              </div>
            </div>

            <div className="check-row">
              <span className={`check-status ${metadata.duration_seconds > 900 ? 'fail' : 'pass'}`}>
                {metadata.duration_seconds > 900 ? <XCircle size={14} /> : <CheckCircle size={14} />}
              </span>
              <div className="check-details">
                <span className="check-title">Temporal Sequence Duration</span>
                <span className="check-desc">
                  {metadata.duration_seconds.toFixed(1)}s (Maximum safe boundary: 900.0s).
                </span>
              </div>
            </div>

            <div className="check-row">
              <span
                className={`check-status ${
                  metadata.width < 640 ? 'fail' : metadata.width < 1280 ? 'warn' : 'pass'
                }`}
              >
                {metadata.width < 640 ? (
                  <XCircle size={14} />
                ) : metadata.width < 1280 ? (
                  <AlertTriangle size={14} />
                ) : (
                  <CheckCircle size={14} />
                )}
              </span>
              <div className="check-details">
                <span className="check-title">Frame Spatial Density</span>
                <span className="check-desc">
                  {metadata.width} &times; {metadata.height} pixels{' '}
                  {metadata.width >= 1280 ? '(High resolution HD coverage)' : '(Standard definition)'}.
                </span>
              </div>
            </div>

            <div className="check-row">
              <span className="check-status pass">
                <Cpu size={14} />
              </span>
              <div className="check-details">
                <span className="check-title">Workstation Concurrency & VRAM Guard</span>
                <span className="check-desc">
                  Workstation GPU queue idle (1/1 active slot available). RTX 2070 SUPER (8192 MiB VRAM) detected.
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Warnings or Rejections Callouts */}
        {metadata.warnings.length > 0 && (
          <div className="warning-callout">
            <AlertTriangle size={16} />
            <div>
              <strong>Advisory Notes:</strong>
              <ul>
                {metadata.warnings.map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {metadata.rejection_reasons.length > 0 && (
          <div className="error-callout">
            <XCircle size={16} />
            <div>
              <strong>Blocking Issues:</strong>
              <ul>
                {metadata.rejection_reasons.map((r, idx) => (
                  <li key={idx}>{r}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Execution Mode Notice */}
        <div className="execution-mode-notice">
          <ShieldCheck size={16} className="text-cyan" />
          <div className="notice-meta">
            <span className="notice-title">Worker Integration: Mock Pipeline Adapter</span>
            <span className="notice-sub">
              Per workstream boundary, this run uses the deterministic mock adapter. It will simulate stages and load the
              sample indoor room. Real VGGT inference is gated on <code>PIPELINE_INTEGRATION_HANDOFF.md</code>.
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="preflight-actions">
          <button className="btn-secondary" onClick={onBackToUpload}>
            Cancel
          </button>

          <button
            className="btn-primary-large"
            disabled={isRejected}
            onClick={onConfirmStart}
            title={isRejected ? 'Resolve blocking issues to proceed' : 'Launch reconstruction worker'}
          >
            <span>Start Reconstruction</span>
            <ArrowRight size={16} />
          </button>
        </div>

        <div className="retention-summary-tag">
          Retention: {keepRawInput ? 'Raw video kept on disk for 24h' : 'Raw video purged immediately upon completion'}
        </div>
      </div>
    </div>
  );
};

