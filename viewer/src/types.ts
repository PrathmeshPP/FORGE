/**
 * F.O.R.G.E. Indoor 3D Reconstruction Platform — Type Definitions
 * Adheres strictly to API_CONTRACT.md, TRD.md, and MODEL_QA_AND_RETRY_POLICY.md
 */

export type JobStatus =
  | 'receiving'
  | 'validating'
  | 'preflight'
  | 'queued'
  | 'running'
  | 'validating_output'
  | 'ready'
  | 'failed'
  | 'cancelling'
  | 'cancelled'
  | 'deleting';

export type PipelineStage =
  | 'idle'
  | 'ingestion'
  | 'keyframe_extraction'
  | 'spatial_inference'
  | 'dense_meshing'
  | 'texturing'
  | 'qa_verification';

export type RepresentationType = 'glb' | 'textured_obj' | 'photographic_ply' | 'untextured_ply';

export interface ModelRepresentation {
  type: 'glb' | 'obj' | 'ply';
  label: string;
  glb_url?: string;
  obj_url?: string;
  mtl_url?: string;
  texture_urls?: string[];
  ply_url?: string;
  file_size_bytes: number;
  sha256?: string;
  available: boolean;
}

export type CameraPresetType = 'top' | 'front' | 'side' | 'iso' | 'reset';

export interface CameraPresetAction {
  type: CameraPresetType;
  nonce: number;
}

export interface ActiveGeometryStats {
  uniqueVertices: number | null;
  renderedPositions: number | null;
  triangles: number | null;
  sourceFileName: string | null;
  format: 'OBJ' | 'PLY' | 'UNKNOWN';
}

export interface BoundingBoxInfo {
  min: [number, number, number];
  max: [number, number, number];
  size: [number, number, number];
  center: [number, number, number];
}

export interface GateReport {
  gate: 'Gate A' | 'Gate B' | 'Gate C' | 'Gate D' | 'Gate E';
  title: string;
  status: 'PASS' | 'WARN' | 'FAIL';
  summary: string;
  details: Record<string, any>;
}

export interface QualitySummary {
  overall_status: 'PASS' | 'PARTIAL' | 'FAIL';
  vertex_count: number;
  triangle_count: number;
  bounding_box_meters?: [number, number, number];
  degenerate_face_ratio: number;
  viewer_load_verified: boolean;
  visual_inspection_passed: boolean;
  known_limitations: string;
  coordinate_system: string;
  gates: GateReport[];
}

export interface SceneEntry {
  id: string;
  display_name: string;
  subtitle: string;
  is_sample_asset: boolean;
  sample_watermark_text?: string;
  acceptance_status: 'PASS' | 'PARTIAL' | 'FAIL';
  known_limitations: string;
  coordinate_status: string;
  vertex_count: number;
  triangle_count: number;
  default_representation: RepresentationType;
  representations: Partial<Record<RepresentationType, ModelRepresentation>>;
  initial_camera: {
    position: [number, number, number];
    target: [number, number, number];
  };
  quality_summary?: QualitySummary;
}

export interface PreflightMetadata {
  filename: string;
  file_size_bytes: number;
  file_size_mb: number;
  mime_type: string;
  container_valid: boolean;
  duration_seconds: number;
  width: number;
  height: number;
  aspect_ratio: string;
  framerate_fps?: number;
  estimated_frames?: number;
  status: 'passed' | 'warning' | 'rejected';
  rejection_reasons: string[];
  warnings: string[];
}

export interface JobRecord {
  id: string;
  display_name: string;
  original_filename: string;
  status: JobStatus;
  progress_stage: PipelineStage;
  progress_stage_label: string;
  worker_mode: 'mock' | 'real';
  is_sample_job: boolean;
  keep_raw_input: boolean;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  attempt_count: number;
  last_error_code?: string;
  last_error_message?: string;
  preflight?: PreflightMetadata;
  scene?: SceneEntry;
  quality_summary?: QualitySummary;
  logs: Array<{ timestamp: string; stage: PipelineStage; message: string }>;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unavailable';
  worker: 'idle' | 'busy' | 'offline';
  accepting_jobs: boolean;
  worker_mode: 'mock' | 'real';
  active_job_id?: string;
  limits: {
    max_upload_bytes: number;
    max_video_duration_seconds: number;
    accepted_extensions: string[];
    max_active_jobs: number;
  };
  host_info?: {
    platform: string;
    gpu_model: string;
    vram_total_mb: number;
    vram_free_mb: number;
    disk_free_gb: number;
  };
}

export type BackgroundTheme = 'slate' | 'charcoal' | 'black' | 'light';

export interface ViewportConfig {
  showGrid: boolean;
  showAxes: boolean;
  wireframe: boolean;
  autoRotate: boolean;
  background: BackgroundTheme;
  brightness: number; // 0.5 to 2.0, default 1.0
}

export interface RecentModelItem {
  id: string;
  scene_name: string;
  source_video_name: string;
  completion_timestamp: string;
  completion_mtime: number;
  reconstruction_status: string;
  is_partial: boolean;
  walkthrough_validated: boolean;
  file_size_bytes: number;
  file_size_mb: number;
  vertex_count: number;
  triangle_count: number;
  floor_continuity_pct: number;
  obstacle_clearance_pct: number;
  glb_url: string;
  thumbnail_url: string;
  has_local_thumbnail: boolean;
}

