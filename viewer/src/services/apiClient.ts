import type {
  JobRecord,
  HealthResponse,
  PreflightMetadata,
  PipelineStage,
} from '../types';
import { SAMPLE_INDOOR_SCENE } from '../data/sampleScene';

const API_BASE = '/api/v1';

// In-memory mock storage for client simulation when backend is offline
let mockJobs: JobRecord[] = [
  {
    id: 'sample-job-0001',
    display_name: 'Living Room Corridor Scan',
    original_filename: 'living_room_scan.mp4',
    status: 'ready',
    progress_stage: 'idle',
    progress_stage_label: 'Reconstruction Ready',
    worker_mode: 'mock',
    is_sample_job: true,
    keep_raw_input: false,
    created_at: new Date(Date.now() - 3600000).toISOString(),
    started_at: new Date(Date.now() - 3580000).toISOString(),
    finished_at: new Date(Date.now() - 3400000).toISOString(),
    duration_seconds: 180,
    attempt_count: 1,
    scene: SAMPLE_INDOOR_SCENE,
    quality_summary: SAMPLE_INDOOR_SCENE.quality_summary,
    logs: [
      { timestamp: '14:00:00', stage: 'ingestion', message: 'Input video verified (1080p, 24 FPS, 45s)' },
      { timestamp: '14:00:05', stage: 'keyframe_extraction', message: 'Extracted 45 sharp keyframes with viewpoint disparity' },
      { timestamp: '14:00:30', stage: 'spatial_inference', message: 'VGGT spatial prediction completed (Peak VRAM: 5.8 GB)' },
      { timestamp: '14:01:45', stage: 'dense_meshing', message: 'Dense point cloud fused and Poisson meshing converged' },
      { timestamp: '14:02:30', stage: 'texturing', message: 'Surface charts projected and texture atlas baked' },
      { timestamp: '14:03:00', stage: 'qa_verification', message: 'Gate A-E checks passed. Bounding box [4.5m x 2.7m x 4.0m]' },
    ],
  },
];

let activeSimulationTimer: any = null;

export const apiClient = {
  async getHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Backend not currently reachable; return deterministic mock health status
    }

    return {
      status: 'ok',
      worker: activeSimulationTimer ? 'busy' : 'idle',
      accepting_jobs: true,
      worker_mode: 'mock',
      limits: {
        max_upload_bytes: 2147483648, // 2 GiB
        max_video_duration_seconds: 900, // 15 mins
        accepted_extensions: ['.mp4', '.mov'],
        max_active_jobs: 1,
      },
      host_info: {
        platform: 'Windows 11 Pro (Build 22631)',
        gpu_model: 'NVIDIA GeForce RTX 2070 SUPER (Detected)',
        vram_total_mb: 8192,
        vram_free_mb: 8100,
        disk_free_gb: 210,
      },
    };
  },

  async listJobs(): Promise<JobRecord[]> {
    try {
      const res = await fetch(`${API_BASE}/jobs`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Fallback to local memory mock
    }
    return [...mockJobs];
  },

  async getJob(id: string): Promise<JobRecord | null> {
    try {
      const res = await fetch(`${API_BASE}/jobs/${id}`, { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Fallback
    }
    const found = mockJobs.find((j) => j.id === id);
    return found ? { ...found } : null;
  },

  async createJob(
    file: File,
    preflight: PreflightMetadata,
    keepRawInput: boolean
  ): Promise<JobRecord> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('keep_raw_input', String(keepRawInput));

      const res = await fetch(`${API_BASE}/jobs`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(30000), // Allow time for large video upload
      });
      if (res.ok) {
        const backendJob = await res.json();
        return backendJob;
      }
    } catch {
      // Fallback to mock job if backend is offline
    }

    const jobId = `job-${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 6)}`;
    const newJob: JobRecord = {
      id: jobId,
      display_name: file.name.replace(/\.[^/.]+$/, ''),
      original_filename: file.name,
      status: 'preflight',
      progress_stage: 'idle',
      progress_stage_label: 'Preflight Passed — Ready to Reconstruct',
      worker_mode: 'mock',
      is_sample_job: true,
      keep_raw_input: keepRawInput,
      created_at: new Date().toISOString(),
      attempt_count: 0,
      preflight,
      logs: [
        {
          timestamp: new Date().toLocaleTimeString(),
          stage: 'ingestion',
          message: `Preflight checks verified: ${preflight.width}x${preflight.height}, ${preflight.duration_seconds.toFixed(1)}s, ${preflight.file_size_mb.toFixed(1)} MB`,
        },
      ],
    };

    mockJobs.unshift(newJob);
    return newJob;
  },

  async startJob(
    jobId: string,
    onProgress?: (updated: JobRecord) => void
  ): Promise<JobRecord> {
    // 1. Attempt Real Backend Worker Start & Live Polling
    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}/start`, {
        method: 'POST',
        signal: AbortSignal.timeout(3000),
      });

      if (res.ok) {
        const startedJob: JobRecord = await res.json();
        if (onProgress) onProgress(startedJob);

        // Start live polling loop against backend
        const pollBackend = async () => {
          try {
            const pollRes = await fetch(`${API_BASE}/jobs/${jobId}`);
            if (pollRes.ok) {
              const currentJob: JobRecord = await pollRes.json();
              if (onProgress) onProgress(currentJob);

              if (['ready', 'failed', 'cancelled'].includes(currentJob.status)) {
                activeSimulationTimer = null;
                return; // Stop polling on terminal states
              }
            }
          } catch {
            // Transient fetch error, retry
          }
          activeSimulationTimer = setTimeout(pollBackend, 800);
        };

        activeSimulationTimer = setTimeout(pollBackend, 400);
        return startedJob;
      }
    } catch {
      // Backend not running; fallback to deterministic in-memory client simulation
    }

    // 2. Offline Simulation Fallback
    const job = mockJobs.find((j) => j.id === jobId);
    if (!job) throw new Error('Job not found');

    job.status = 'queued';
    job.progress_stage = 'ingestion';
    job.progress_stage_label = 'Waiting in Workstation Task Queue...';
    job.started_at = new Date().toISOString();
    job.attempt_count = 1;
    if (onProgress) onProgress({ ...job });

    const stages: Array<{
      status: JobRecord['status'];
      stage: PipelineStage;
      label: string;
      delayMs: number;
      log: string;
    }> = [
        {
          status: 'running',
          stage: 'ingestion',
          label: 'Step 1/5: Streaming Video Ingestion & Container Integrity Check',
          delayMs: 1000,
          log: 'Container parsed without errors. Audio track discarded; video stream validated.',
        },
        {
          status: 'running',
          stage: 'keyframe_extraction',
          label: 'Step 2/5: Viewpoint Diversity & Laplacian Sharpness Curation',
          delayMs: 1500,
          log: 'Extracted keyframes. Mean Laplacian sharpness: 34.2 (threshold: 15.0). Viewpoint baselines confirmed.',
        },
        {
          status: 'running',
          stage: 'spatial_inference',
          label: 'Step 3/5: Foundation Spatial Inference (Simulated Model Pass)',
          delayMs: 2000,
          log: '[MOCK ADAPTER] Simulated VGGT-Omega spatial inference pass. Estimated VRAM ceiling: 6.2 GB.',
        },
        {
          status: 'running',
          stage: 'dense_meshing',
          label: 'Step 4/5: Point Cloud Densification & Surface Reconstruction',
          delayMs: 1800,
          log: 'Dense point cloud generated. Screened Poisson surface reconstruction converged (Depth: 9).',
        },
        {
          status: 'validating_output',
          stage: 'qa_verification',
          label: 'Step 5/5: Running Gates A-E Quality & Coordinate Verification',
          delayMs: 1200,
          log: 'Gate A-E diagnostics executed. Bounding box extents valid. Zero non-finite coordinates.',
        },
      ];

    let currentStep = 0;
    const runNextStep = () => {
      if (currentStep < stages.length) {
        const step = stages[currentStep];
        job.status = step.status;
        job.progress_stage = step.stage;
        job.progress_stage_label = step.label;
        job.logs.push({
          timestamp: new Date().toLocaleTimeString(),
          stage: step.stage,
          message: step.log,
        });
        if (onProgress) onProgress({ ...job });
        currentStep++;
        activeSimulationTimer = setTimeout(runNextStep, step.delayMs);
      } else {
        job.status = 'ready';
        job.progress_stage = 'idle';
        job.progress_stage_label = 'Reconstruction Complete';
        job.finished_at = new Date().toISOString();
        job.duration_seconds = Math.round(
          (new Date(job.finished_at).getTime() - new Date(job.started_at!).getTime()) / 1000
        );
        job.scene = {
          ...SAMPLE_INDOOR_SCENE,
          id: `scene-${job.id}`,
          display_name: `${job.display_name} (Sample Result)`,
          is_sample_asset: true,
          sample_watermark_text:
            '⚠️ MOCK PIPELINE OUTPUT — Visual development placeholder only. Not actual video result.',
        };
        job.quality_summary = SAMPLE_INDOOR_SCENE.quality_summary;
        job.logs.push({
          timestamp: new Date().toLocaleTimeString(),
          stage: 'idle',
          message:
            'Reconstruction ready for inspection. (Note: Mock pipeline adapter used; real pipeline integration pending).',
        });
        activeSimulationTimer = null;
        if (onProgress) onProgress({ ...job });
      }
    };

    activeSimulationTimer = setTimeout(runNextStep, 400);
    return job;
  },

  async cancelJob(jobId: string): Promise<boolean> {
    if (activeSimulationTimer) {
      clearTimeout(activeSimulationTimer);
      activeSimulationTimer = null;
    }

    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}/cancel`, { method: 'POST' });
      if (res.ok) {
        return true;
      }
    } catch {
      // Mock cancellation fallback
    }

    const job = mockJobs.find((j) => j.id === jobId);
    if (job) {
      job.status = 'cancelled';
      job.progress_stage = 'idle';
      job.progress_stage_label = 'Job Cancelled by User';
      job.logs.push({
        timestamp: new Date().toLocaleTimeString(),
        stage: 'idle',
        message: 'Cancellation requested. Workstation child processes safely terminated.',
      });
      return true;
    }
    return false;
  },

  async deleteJob(jobId: string): Promise<boolean> {
    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' });
      if (res.ok) {
        return true;
      }
    } catch {
      // Mock delete
    }

    const initialLen = mockJobs.length;
    mockJobs = mockJobs.filter((j) => j.id !== jobId);
    return mockJobs.length < initialLen;
  },
};
