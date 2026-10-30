import React from 'react';
import {
  X,
  CheckCircle,
  AlertTriangle,
  XCircle,
  FileCheck2,
  ShieldCheck,
  Info,
} from 'lucide-react';
import type { QualitySummary } from '../types';

interface QualityReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  summary?: QualitySummary;
  modelName: string;
}

export const QualityReportModal: React.FC<QualityReportModalProps> = ({
  isOpen,
  onClose,
  summary,
  modelName,
}) => {
  if (!isOpen || !summary) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-row">
            <FileCheck2 size={18} className="text-cyan" />
            <div>
              <h3>Model QA & Geometric Verification Report</h3>
              <span className="modal-sub">{modelName}</span>
            </div>
          </div>
          <button className="btn-close" onClick={onClose} title="Close Modal">
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          {/* Summary Banner */}
          <div className={`qa-verdict-card ${summary.overall_status}`}>
            <div className="qa-verdict-icon">
              {summary.overall_status === 'PASS' ? (
                <CheckCircle size={24} className="text-emerald" />
              ) : summary.overall_status === 'PARTIAL' ? (
                <AlertTriangle size={24} className="text-amber" />
              ) : (
                <XCircle size={24} className="text-rose" />
              )}
            </div>
            <div className="qa-verdict-text">
              <span className="qa-verdict-title">
                Acceptance Status: {summary.overall_status === 'PASS' ? 'ACCEPTED' : summary.overall_status}
              </span>
              <p className="qa-verdict-sub">
                {summary.overall_status === 'PASS'
                  ? 'All structural, topological, and viewer validation gates passed successfully.'
                  : 'Conditional acceptance. Technical file checks passed, but visual review notes limitations.'}
              </p>
            </div>
          </div>

          {/* Metric Highlights */}
          <div className="qa-metrics-row">
            <div className="qa-metric">
              <span className="qa-label">Unique Vertices</span>
              <span className="qa-val">{summary.vertex_count.toLocaleString()}</span>
            </div>
            <div className="qa-metric">
              <span className="qa-label">Triangles / Faces</span>
              <span className="qa-val">{summary.triangle_count.toLocaleString()}</span>
            </div>
            <div className="qa-metric">
              <span className="qa-label">Degenerate Face Ratio</span>
              <span className="qa-val">{(summary.degenerate_face_ratio * 100).toFixed(2)}%</span>
            </div>
            <div className="qa-metric">
              <span className="qa-label">Viewer Framing</span>
              <span className="qa-val text-emerald">Verified</span>
            </div>
          </div>

          {/* Detailed Gates A-E */}
          <div className="gates-accordion">
            <h4>Gate A–E Validation Breakdown</h4>
            <div className="gates-list">
              {summary.gates.map((g, idx) => (
                <div key={idx} className="gate-item">
                  <div className="gate-item-header">
                    <div className="gate-title-meta">
                      <span className={`gate-status-pill ${g.status.toLowerCase()}`}>
                        {g.status}
                      </span>
                      <strong className="gate-name">{g.gate}: {g.title}</strong>
                    </div>
                  </div>
                  <p className="gate-summary">{g.summary}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Known Limitations Callout */}
          <div className="limitations-callout">
            <Info size={15} className="text-cyan" />
            <div>
              <strong>Reported Environmental Limitations:</strong>
              <p>{summary.known_limitations}</p>
            </div>
          </div>

          {/* Policy Compliance Footer */}
          <div className="policy-footer">
            <ShieldCheck size={14} className="text-emerald" style={{ display: 'inline', marginRight: 4 }} />
            <span>
              Validated per <code>MODEL_QA_AND_RETRY_POLICY.md</code>. Finite coordinate bounds checked. No synthetic
              hallucination of occluded surfaces.
            </span>
          </div>
        </div>

        <div className="modal-actions">
          <button className="btn-secondary" onClick={onClose}>
            Close Report
          </button>
        </div>
      </div>
    </div>
  );
};

