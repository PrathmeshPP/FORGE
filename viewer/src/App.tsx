import React, { useState, useEffect, useCallback } from 'react';
import { NavigationViewer } from './components/NavigationViewer';
import { Minimap } from './components/Minimap';
import { ControlsFooter } from './components/ControlsFooter';
import { TopBar, type ModelOption } from './components/TopBar';

export const App: React.FC = () => {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelOption | null>(null);
  const [localFile, setLocalFile] = useState<File | null>(null);

  // Minimap state
  const [cameraPos, setCameraPos] = useState<{ x: number; y: number; z: number }>({ x: 0, y: 1.2, z: 2.5 });
  const [cameraYaw, setCameraYaw] = useState<number>(0);
  const [modelBounds, setModelBounds] = useState<{ minX: number; maxX: number; minZ: number; maxZ: number } | null>(null);
  const [footprintPoints, setFootprintPoints] = useState<{ x: number; z: number }[]>([]);

  // Loading state
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingProgress, setLoadingProgress] = useState<number>(0);
  const [resetTrigger, setResetTrigger] = useState<number>(0);

  // Drag and drop state
  const [isDraggingFile, setIsDraggingFile] = useState<boolean>(false);

  // 1. Fetch available qualifying models stored on PC from backend API
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/v1/recent-models?limit=10');
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data) && data.length > 0) {
            const formatted: ModelOption[] = data.map((item: any) => ({
              id: item.id,
              name: formatModelName(item.id, item.scene_name),
              url: item.glb_url || `/api/v1/models/${item.id}/glb`,
              sizeMb: item.file_size_mb,
              source: 'server',
            }));
            setModels(formatted);
            setSelectedModel(formatted[0]);
            return;
          }
        }
      } catch (err) {
        console.warn('Could not connect to /api/v1/recent-models:', err);
      }

      // Default fallback option
      const fallback: ModelOption = {
        id: 'sample-viewer-fixture',
        name: 'Drop any .GLB model from PC',
        url: '/sample/sample_indoor_fixture.glb',
        sizeMb: 5.2,
        source: 'server',
      };
      setModels([fallback]);
      setSelectedModel(fallback);
    };

    fetchModels();
  }, []);

  const formatModelName = (id: string, sceneName?: string): string => {
    if (sceneName && sceneName.length > 2) return sceneName;
    if (id.includes('improved_b314_floor_leveled')) return 'Hostel Room B-314 (Floor Leveled)';
    if (id.includes('improved_lab_tsdf_flattened')) return 'RRU Computer Lab (TSDF Flattened)';
    if (id.includes('candidate_b314')) return 'Room B-314 (Candidate Boundary)';
    if (id.includes('candidate_lab')) return 'Computer Lab (Candidate Boundary)';
    if (id.includes('improved_quality_b314')) return 'Room B-314 (Improved Baseline)';
    if (id.includes('improved_quality_lab')) return 'Computer Lab (Improved Baseline)';
    if (id.includes('floor_fixed_b314')) return 'Room B-314 (Floor Fixed)';
    if (id.includes('floor_fixed_corridor')) return 'Corridor (Floor Fixed)';
    if (id.includes('floor_fixed_nov21')) return 'Multi-Room Living Room (Floor Fixed)';
    return id.replace(/_/g, ' ');
  };

  // 2. Select Model from Dropdown
  const handleSelectModel = (model: ModelOption) => {
    setLocalFile(null);
    setSelectedModel(model);
  };

  // 3. Upload / Select Local File from PC
  const handleUploadLocalFile = (file: File) => {
    const localOption: ModelOption = {
      id: `local-${file.name}-${Date.now()}`,
      name: `📁 ${file.name}`,
      url: '',
      sizeMb: Math.round((file.size / (1024 * 1024)) * 10) / 10,
      source: 'local',
    };
    setModels((prev) => [localOption, ...prev]);
    setSelectedModel(localOption);
    setLocalFile(file);
  };

  // 4. Drag & Drop File Handling
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingFile(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingFile(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingFile(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.toLowerCase().endsWith('.glb') || file.name.toLowerCase().endsWith('.gltf')) {
        handleUploadLocalFile(file);
      }
    }
  }, []);

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{
        width: '100vw',
        height: '100vh',
        position: 'relative',
        overflow: 'hidden',
        background: '#090d16',
        margin: 0,
        padding: 0,
      }}
    >
      {/* Floating Top Bar (Model Picker & Open File from PC) */}
      <TopBar
        models={models}
        selectedModelId={selectedModel?.id || ''}
        onSelectModel={handleSelectModel}
        onUploadLocalFile={handleUploadLocalFile}
        onResetCamera={() => setResetTrigger((prev) => prev + 1)}
        isLoading={isLoading}
        loadingProgress={loadingProgress}
      />

      {/* Minimap in the Right Corner */}
      <Minimap
        cameraPos={cameraPos}
        cameraYaw={cameraYaw}
        bounds={modelBounds}
        footprintPoints={footprintPoints}
      />

      {/* Main Full-Screen 3D Navigation Viewport */}
      <NavigationViewer
        modelUrl={localFile ? null : selectedModel?.url || null}
        modelFile={localFile}
        onCameraUpdate={(pos, yaw) => {
          setCameraPos(pos);
          setCameraYaw(yaw);
        }}
        onBoundsUpdate={(bounds, pts) => {
          setModelBounds(bounds);
          setFootprintPoints(pts);
        }}
        onLoadingChange={(loading, pct) => {
          setIsLoading(loading);
          setLoadingProgress(pct);
        }}
        resetTrigger={resetTrigger}
      />

      {/* Navigation Controls Legend (Explicitly written at bottom of website) */}
      <ControlsFooter />

      {/* Drag & Drop Visual Overlay */}
      {isDraggingFile && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(2, 132, 199, 0.4)',
            border: '3px dashed #38bdf8',
            backdropFilter: 'blur(4px)',
            zIndex: 100,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#f8fafc',
            fontFamily: 'system-ui, sans-serif',
            fontSize: '20px',
            fontWeight: 700,
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontSize: '48px', marginBottom: '12px' }}>📥</div>
          Drop .GLB file here to navigate 3D model!
        </div>
      )}
    </div>
  );
};
