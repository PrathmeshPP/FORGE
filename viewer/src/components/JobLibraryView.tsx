import React from 'react';
import {
  Layers,
  Box,
  Trash2,
  Clock,
  HardDrive,
  Info,
  CheckCircle,
  Activity,
  XCircle,
  StopCircle,
} from 'lucide-react';
import type { JobRecord } from '../types';

interface JobLibraryViewProps {
  jobs: JobRecord[];
  onSelectJob: (job: JobRecord) => void;
  onDeleteJob: (jobId: string) => void;
}

export const JobLibraryView: React.FC<JobLibraryViewProps> = ({
  jobs,
  onSelectJob,
  onDeleteJob,
}) => {
  const formatTime = (secs?: number) => {
    if (!secs) return '—';
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}m ${s < 10 ? '0' : ''}${s}s`;
  };

  const formatDate = (iso: string) => {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return iso;
    }
  };

  return (
    <div className="library-view-container">
      <div className="library-card">
        <div className="library-header">
          <div>
            <h2>Reconstruction Job Library</h2>
            <p className="subtitle">
              Local workstation queue and completed indoor digital twins.
            </p>
          </div>
        </div>

        {/* Visibility disclosure banner */}
        <div className="privacy-disclosure">
          <Info size={15} className="text-cyan" />
          <span>
            <strong>Private Team Visibility (v1):</strong> All collaborators authorized through the host's Tailscale
            Serve policy share visibility of this local job library.
          </span>
        </div>

        {jobs.length === 0 ? (
          <div className="empty-library">
            <Layers size={36} className="text-muted" />
            <p>No reconstruction jobs recorded yet.</p>
            <span className="empty-sub">Submit an indoor smartphone video or load a sample fixture.</span>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="jobs-table">
              <thead>
                <tr>
                  <th>Job Name & ID</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Duration</th>
                  <th>Input File</th>
                  <th>Worker Mode</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} className="job-row">
                    <td>
                      <div className="job-name-cell">
                        <strong className="job-name">{j.display_name}</strong>
                        <span className="job-id mono">{j.id}</span>
                      </div>
                    </td>
                    <td>
                      <span className={`status-pill ${j.status}`}>
                        {j.status === 'running' && <Activity size={11} className="pulse-icon" />}
                        {j.status === 'ready' && <CheckCircle size={11} />}
                        {j.status === 'failed' && <XCircle size={11} />}
                        {j.status === 'cancelled' && <StopCircle size={11} />}
                        <span>{j.status}</span>
                      </span>
                    </td>
                    <td>
                      <span className="table-meta">{formatDate(j.created_at)}</span>
                    </td>
                    <td>
                      <span className="table-meta">
                        <Clock size={11} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
                        {formatTime(j.duration_seconds)}
                      </span>
                    </td>
                    <td>
                      <span className="table-meta truncate" title={j.original_filename}>
                        <HardDrive size={11} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
                        {j.original_filename}
                      </span>
                    </td>
                    <td>
                      <span className="mode-tag">{j.worker_mode.toUpperCase()}</span>
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div className="action-buttons">
                        {j.status === 'ready' && (
                          <button
                            className="btn-table-action"
                            onClick={() => onSelectJob(j)}
                            title="Open 3D Model in Viewer"
                          >
                            <Box size={13} />
                            <span>View 3D</span>
                          </button>
                        )}
                        <button
                          className="btn-table-danger"
                          onClick={() => {
                            if (confirm(`Delete job "${j.display_name}" and remove local assets?`)) {
                              onDeleteJob(j.id);
                            }
                          }}
                          title="Delete Job"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

