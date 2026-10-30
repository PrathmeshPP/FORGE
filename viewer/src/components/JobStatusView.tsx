import React, { useEffect, useState } from 'react';
import {
  Activity,
  CheckCircle,
  XCircle,
  Clock,
  Cpu,
  StopCircle,
  Box,
  Terminal,
  ShieldAlert,
  ArrowRight,
} from 'lucide-react';
import type { JobRecord, PipelineStage } from '../types';

interface JobStatusViewProps {
  job: JobRecord;
  onCancelJob: (jobId: string) => void;
  onOpenViewer: () => void;
  onStartNewUpload: () => void;
}

export const JobStatusView: React.FC<JobStatusViewProps> = ({
  job,
  onCancelJob,
  onOpenViewer,
  onStartNewUpload,
}) => {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (job.status === 'running' || job.status === 'queued' || job.status === 'validating_output') {
      const start = job.started_at ? new Date(job.started_at).getTime() : Date.now();
      const interval = setInterval(() => {
        setElapsedSeconds(Math.max(0, Math.floor((Date.now() - start) / 1000)));
      }, 1000);
      return () => clearInterval(interval);
    } else if (job.duration_seconds) {
      setElapsedSeconds(job.duration_seconds);
    }
  }, [job.status, job.started_at, job.duration_seconds]);

  const stages: Array<{ id: PipelineStage; name: string; desc: string }> = [
    { id: 'ingestion', name: 'Ingestion & Preflight', desc: 'Container decodability and disk workspace setup' },
    { id: 'keyframe_extraction', name: 'Keyframe Extraction', desc: 'Laplacian sharpness and viewpoint diversity analysis' },
    { id: 'spatial_inference', name: 'Foundation Model Inference', desc: 'VGGT spatial prediction & camera trajectory decoding' },
    { id: 'dense_meshing', name: 'Dense Fusion & Meshing', desc: 'Point cloud densification & surface mesh reconstruction' },
    { id: 'texturing', name: 'Texture & Materials', desc: 'Surface chart projection & texture atlas export' },
    { id: 'qa_verification', name: 'Gate A-E Validation', desc: 'Finite geometry checks & viewer compatibility verification' },
  ];

  const getStageState = (stageId: PipelineStage) => {
    if (job.status === 'ready') return 'completed';
    if (job.status === 'failed' || job.status === 'cancelled') {
      if (job.progress_stage === stageId) return 'failed';
    }

    const order: PipelineStage[] = [
      'ingestion',
      'keyframe_extraction',
      'spatial_inference',
      'dense_meshing',
      'texturing',
      'qa_verification',
    ];
    const currentIndex = order.indexOf(job.progress_stage);
    const targetIndex = order.indexOf(stageId);

    if (currentIndex > targetIndex) return 'completed';
    if (currentIndex === targetIndex) return 'active';
    return 'pending';
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}m ${s < 10 ? '0' : ''}${s}s`;
  };

  return (
    <div className="status-view-container">
      <div className="status-card">
        {/* Header */}
        <div className="status-header">
          <div className="status-title-row">
            <div>
              <span className="mono-job-id">JOB ID: {job.id}</span>
              <h2>{job.display_name}</h2>
            </div>
            <div className={`status-badge-large ${job.status}`}>
              {job.status === 'running' && <Activity size={16} className="pulse-icon" />}
              {job.status === 'ready' && <CheckCircle size={16} />}
              {job.status === 'failed' && <XCircle size={16} />}
              {job.status === 'cancelled' && <StopCircle size={16} />}
              <span>{job.status.toUpperCase()}</span>
            </div>
          </div>

          {/* Banner if mock adapter */}
          <div className="mock-banner">
            <ShieldAlert size={14} className="text-amber" />
            <span>
              <strong>MOCK WORKER EXECUTION:</strong> Simulated pipeline adapter is running. Real model inference will not
              execute until <code>PIPELINE_INTEGRATION_HANDOFF.md</code> is approved.
            </span>
          </div>

          {/* Error Banner if job failed */}
          {job.status === 'failed' && (
            <div className="status-error-banner">
              <XCircle size={16} className="text-rose" />
              <div>
                <strong>Reconstruction Error ({job.last_error_code || 'PIPELINE_ERROR'}):</strong>
                <p>{job.last_error_message || 'Reconstruction process failed. Review log entries below.'}</p>
              </div>
            </div>
          )}

          {/* Cancellation Banner if job cancelled */}
          {job.status === 'cancelled' && (
            <div className="status-cancelled-banner">
              <StopCircle size={16} />
              <div>
                <strong>Reconstruction Cancelled:</strong>
                <p>Execution halted upon user cancellation request. Subprocess stopped.</p>
              </div>
            </div>
          )}
        </div>

        {/* Live Metrics Row */}
        <div className="metrics-row">
          <div className="metric-box">
            <span className="metric-label">
              <Clock size={12} /> Elapsed Time
            </span>
            <span className="metric-value mono">{formatTime(elapsedSeconds)}</span>
          </div>

          <div className="metric-box">
            <span className="metric-label">
              <Cpu size={12} /> Workstation Target
            </span>
            <span className="metric-value">NVIDIA RTX 2070 SUPER</span>
          </div>

          <div className="metric-box">
            <span className="metric-label">
              <Activity size={12} /> Current Stage
            </span>
            <span className="metric-value truncate">{job.progress_stage_label || 'Processing...'}</span>
          </div>
        </div>

        {/* Stage Progression Checklist (Honest, no fake percentages) */}
        <div className="stages-flow">
          <h3>Reconstruction Execution Stages</h3>
          <div className="stages-list">
            {stages.map((st) => {
              const state = getStageState(st.id);
              return (
                <div key={st.id} className={`stage-step ${state}`}>
                  <div className="step-indicator">
                    {state === 'completed' && <CheckCircle size={16} className="text-emerald" />}
                    {state === 'active' && <div className="step-spinner" />}
                    {state === 'pending' && <span className="step-dot" />}
                    {state === 'failed' && <XCircle size={16} className="text-rose" />}
                  </div>
                  <div className="step-content">
                    <span className="step-name">{st.name}</span>
                    <span className="step-desc">{st.desc}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Live Logs Terminal Window */}
        <div className="terminal-container">
          <div className="terminal-header">
            <Terminal size={13} />
            <span>Workstation Pipeline Event Stream</span>
          </div>
          <div className="terminal-body">
            {job.logs.length === 0 ? (
              <div className="log-empty">Waiting for subprocess event stream...</div>
            ) : (
              job.logs.map((log, idx) => (
                <div key={idx} className="log-line">
                  <span className="log-time">[{log.timestamp}]</span>
                  <span className="log-stage">[{log.stage}]</span>
                  <span className="log-msg">{log.message}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Action Controls */}
        <div className="status-actions">
          {['running', 'queued', 'validating_output'].includes(job.status) ? (
            <button className="btn-danger" onClick={() => onCancelJob(job.id)}>
              <StopCircle size={15} />
              <span>Cancel Job Execution</span>
            </button>
          ) : null}

          {job.status === 'ready' && (
            <button className="btn-primary-large" onClick={onOpenViewer}>
              <Box size={16} />
              <span>Inspect 3D Digital Twin</span>
              <ArrowRight size={16} />
            </button>
          )}

          {job.status === 'failed' || job.status === 'cancelled' ? (
            <button className="btn-secondary" onClick={onStartNewUpload}>
              Start New Upload
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
};

