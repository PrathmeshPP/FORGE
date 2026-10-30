import React, { useState, useEffect, useCallback } from 'react';
import type {
  ActiveTab,
} from './components/Header';
import { Header } from './components/Header';
import { UploadView } from './components/UploadView';
import { PreflightView } from './components/PreflightView';
import { JobStatusView } from './components/JobStatusView';
import { Viewport3D } from './components/Viewport3D';
import { Toolbar } from './components/Toolbar';
import { GeometryStatsDisplay } from './components/GeometryStatsDisplay';
import { QualityReportModal } from './components/QualityReportModal';
import { JobLibraryView } from './components/JobLibraryView';
import { SystemDiagnosticsView } from './components/SystemDiagnosticsView';

import { apiClient } from './services/apiClient';
import { SAMPLE_INDOOR_SCENE } from './data/sampleScene';
import type {
  JobRecord,
  HealthResponse,
  PreflightMetadata,
  SceneEntry,
  RepresentationType,
  ViewportConfig,
  ActiveGeometryStats,
  BoundingBoxInfo,
  CameraPresetAction,
  CameraPresetType,
} from './types';

export const App: React.FC = () => {
  const getInitialTab = (): ActiveTab => {
    try {
      const params = new URLSearchParams(window.location.search);
      const qTab = params.get('tab') as ActiveTab;
      if (['upload', 'preflight', 'status', 'viewer', 'library', 'diagnostics'].includes(qTab)) {
        return qTab;
      }
      const hash = window.location.hash.replace('#', '') as ActiveTab;
      if (['upload', 'preflight', 'status', 'viewer', 'library', 'diagnostics'].includes(hash)) {
        return hash;
      }
    } catch {
      // Fallback
    }
    return 'upload';
  };

  const [activeTab, setActiveTabState] = useState<ActiveTab>(getInitialTab);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const setActiveTab = (tab: ActiveTab) => {
    window.location.hash = tab;
    setActiveTabState(tab);
  };

  useEffect(() => {
    const onHashChange = () => {
      const h = window.location.hash.replace('#', '') as ActiveTab;
      if (['upload', 'preflight', 'status', 'viewer', 'library', 'diagnostics'].includes(h)) {
        setActiveTabState(h);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  // Upload & Preflight State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preflightData, setPreflightData] = useState<PreflightMetadata | null>(null);
  const [keepRawInput, setKeepRawInput] = useState(false);

  // Active Job State
  const [activeJob, setActiveJob] = useState<JobRecord | null>(null);
  const [allJobs, setAllJobs] = useState<JobRecord[]>([
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
  ]);

  // 3D Viewer State
  const [currentScene, setCurrentScene] = useState<SceneEntry>(SAMPLE_INDOOR_SCENE);
  const [currentRepresentation, setCurrentRepresentation] = useState<RepresentationType>('textured_obj');
  const [viewportConfig, setViewportConfig] = useState<ViewportConfig>({
    showGrid: true,
    showAxes: false,
    wireframe: false,
    autoRotate: false,
    background: 'slate',
  });
  const [presetAction, setPresetAction] = useState<CameraPresetAction | undefined>(undefined);
  const [activeStats, setActiveStats] = useState<ActiveGeometryStats | null>(null);
  const [boundingBox, setBoundingBox] = useState<BoundingBoxInfo | null>(null);
  const [isQualityModalOpen, setIsQualityModalOpen] = useState(false);

  // Poll Health and Job list on mount
  const refreshState = useCallback(async () => {
    try {
      const h = await apiClient.getHealth();
      setHealth(h);
      const jobs = await apiClient.listJobs();
      setAllJobs(jobs);
    } catch {
      // Offline fallback is handled gracefully
    }
  }, []);

  useEffect(() => {
    refreshState();
    const interval = setInterval(refreshState, 5000);
    return () => clearInterval(interval);
  }, [refreshState]);

  // Handler: When user selects a video file for preflight
  const handleVideoSelected = (file: File, preflight: PreflightMetadata, keepRaw: boolean) => {
    setSelectedFile(file);
    setPreflightData(preflight);
    setKeepRawInput(keepRaw);
    setActiveTab('preflight');
  };

  // Handler: Start reconstruction from preflight
  const handleStartReconstruction = async () => {
    if (!selectedFile || !preflightData) return;

    try {
      const job = await apiClient.createJob(selectedFile, preflightData, keepRawInput);
      setActiveJob(job);
      setActiveTab('status');

      // Begin execution
      apiClient.startJob(job.id, (updated) => {
        setActiveJob({ ...updated });
        if (updated.status === 'ready' && updated.scene) {
          setCurrentScene(updated.scene);
        }
        refreshState();
      });
    } catch (err: any) {
      alert(`Error starting reconstruction: ${err.message}`);
    }
  };

  // Handler: Load sample model directly
  const handleLoadSampleModel = () => {
    setCurrentScene(SAMPLE_INDOOR_SCENE);
    setCurrentRepresentation('textured_obj');
    setActiveTab('viewer');
  };

  // Handler: Cancel active job
  const handleCancelJob = async (jobId: string) => {
    if (confirm('Cancel running reconstruction? Workstation subprocesses will be terminated.')) {
      await apiClient.cancelJob(jobId);
      const updated = await apiClient.getJob(jobId);
      if (updated) setActiveJob(updated);
      refreshState();
    }
  };

  // Handler: Select job from library to view
  const handleSelectJobFromLibrary = (job: JobRecord) => {
    if (job.scene) {
      setCurrentScene(job.scene);
      setActiveTab('viewer');
    } else {
      setActiveJob(job);
      setActiveTab('status');
    }
  };

  // Handler: Delete job
  const handleDeleteJob = async (jobId: string) => {
    await apiClient.deleteJob(jobId);
    if (activeJob?.id === jobId) setActiveJob(null);
    refreshState();
  };

  // Preset camera action helper
  const handlePresetCamera = (type: CameraPresetType) => {
    setPresetAction({ type, nonce: Date.now() });
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        health={health}
        activeJobId={activeJob && activeJob.status === 'running' ? activeJob.id : null}
        hasReadyResult={currentScene !== null}
      />

      <main className="main-content">
        {/* TAB 1: UPLOAD */}
        {activeTab === 'upload' && (
          <UploadView
            onVideoSelected={handleVideoSelected}
            onLoadSampleModel={handleLoadSampleModel}
            maxUploadBytes={health?.limits.max_upload_bytes || 2147483648}
            acceptedExtensions={health?.limits.accepted_extensions || ['.mp4', '.mov']}
          />
        )}

        {/* TAB 2: PREFLIGHT */}
        {activeTab === 'preflight' && preflightData && (
          <PreflightView
            metadata={preflightData}
            keepRawInput={keepRawInput}
            onConfirmStart={handleStartReconstruction}
            onBackToUpload={() => setActiveTab('upload')}
          />
        )}

        {/* TAB 3: PROCESSING / JOB STATUS */}
        {activeTab === 'status' && (activeJob || allJobs[0]) && (
          <JobStatusView
            job={activeJob || allJobs[0]}
            onCancelJob={handleCancelJob}
            onOpenViewer={() => setActiveTab('viewer')}
            onStartNewUpload={() => setActiveTab('upload')}
          />
        )}

        {/* TAB 4: 3D VIEWPORT */}
        {activeTab === 'viewer' && (
          <div className="viewer-layout">
            <Toolbar
              scene={currentScene}
              currentRepresentation={currentRepresentation}
              onSelectRepresentation={setCurrentRepresentation}
              config={viewportConfig}
              onUpdateConfig={(up) => setViewportConfig((prev) => ({ ...prev, ...up }))}
              onViewPreset={handlePresetCamera}
              onResetCamera={() => handlePresetCamera('reset')}
              onOpenQualityModal={() => setIsQualityModalOpen(true)}
            />

            <div className="viewport-stage">
              <Viewport3D
                scene={currentScene}
                representation={currentRepresentation}
                config={viewportConfig}
                onLoadingChange={() => { }}
                onBoundingBoxChange={setBoundingBox}
                onStatsChange={setActiveStats}
                presetAction={presetAction}
              />

              <GeometryStatsDisplay stats={activeStats} boundingBox={boundingBox} />
            </div>

            <QualityReportModal
              isOpen={isQualityModalOpen}
              onClose={() => setIsQualityModalOpen(false)}
              summary={currentScene.quality_summary}
              modelName={currentScene.display_name}
            />
          </div>
        )}

        {/* TAB 5: JOB LIBRARY */}
        {activeTab === 'library' && (
          <JobLibraryView
            jobs={allJobs}
            onSelectJob={handleSelectJobFromLibrary}
            onDeleteJob={handleDeleteJob}
          />
        )}

        {/* TAB 6: DIAGNOSTICS */}
        {activeTab === 'diagnostics' && <SystemDiagnosticsView health={health} />}
      </main>
    </div>
  );
};

