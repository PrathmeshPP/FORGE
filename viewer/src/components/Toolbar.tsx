import React from 'react';
import {
  Grid,
  Compass,
  RotateCcw,
  Eye,
  FileCheck2,
  Download,
} from 'lucide-react';
import type {
  SceneEntry,
  RepresentationType,
  ViewportConfig,
  BackgroundTheme,
  CameraPresetType,
} from '../types';

interface ToolbarProps {
  scene: SceneEntry;
  currentRepresentation: RepresentationType;
  onSelectRepresentation: (rep: RepresentationType) => void;
  config: ViewportConfig;
  onUpdateConfig: (newConfig: Partial<ViewportConfig>) => void;
  onViewPreset: (preset: CameraPresetType) => void;
  onResetCamera: () => void;
  onOpenQualityModal: () => void;
}

export const Toolbar: React.FC<ToolbarProps> = ({
  scene,
  currentRepresentation,
  onSelectRepresentation,
  config,
  onUpdateConfig,
  onViewPreset,
  onResetCamera,
  onOpenQualityModal,
}) => {
  const bgThemes: { key: BackgroundTheme; label: string }[] = [
    { key: 'slate', label: 'Dark Slate' },
    { key: 'charcoal', label: 'Charcoal' },
    { key: 'black', label: 'Pitch Black' },
    { key: 'light', label: 'Light Studio' },
  ];

  const handleDownload = () => {
    const rep = scene.representations[currentRepresentation];
    const url = rep?.obj_url || rep?.ply_url;
    if (url) {
      const a = document.createElement('a');
      a.href = url;
      a.download = url.split('/').pop() || 'model';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="viewport-toolbar">
      {/* Representation Selector */}
      <div className="toolbar-group">
        {(Object.keys(scene.representations) as RepresentationType[]).map((repKey) => {
          const rep = scene.representations[repKey];
          if (!rep || !rep.available) return null;
          const isActive = currentRepresentation === repKey;
          return (
            <button
              key={repKey}
              className={`toolbar-btn ${isActive ? 'active' : ''}`}
              onClick={() => onSelectRepresentation(repKey)}
              title={`Switch 3D representation to ${rep.label}`}
            >
              <Eye size={13} />
              <span>
                {repKey === 'textured_obj'
                  ? 'Textured OBJ'
                  : repKey === 'photographic_ply'
                    ? 'Photo PLY'
                    : 'Raw PLY'}
              </span>
            </button>
          );
        })}
      </div>

      <div className="toolbar-separator" />

      {/* Camera Presets */}
      <div className="toolbar-group">
        <button className="toolbar-btn" onClick={() => onViewPreset('top')} title="Top Overhead Elevation">
          Top
        </button>
        <button className="toolbar-btn" onClick={() => onViewPreset('front')} title="Front Elevation">
          Front
        </button>
        <button className="toolbar-btn" onClick={() => onViewPreset('side')} title="Side Profile">
          Side
        </button>
        <button className="toolbar-btn" onClick={() => onViewPreset('iso')} title="Isometric Perspective">
          Iso
        </button>
        <button className="toolbar-btn icon-only" onClick={onResetCamera} title="Reset Camera Framing">
          <RotateCcw size={13} />
        </button>
      </div>

      <div className="toolbar-separator" />

      {/* Viewport Toggles */}
      <div className="toolbar-group">
        <button
          className={`toolbar-btn icon-only ${config.wireframe ? 'active' : ''}`}
          onClick={() => onUpdateConfig({ wireframe: !config.wireframe })}
          title="Toggle Wireframe Mesh"
        >
          <Compass size={13} />
        </button>
        <button
          className={`toolbar-btn icon-only ${config.showGrid ? 'active' : ''}`}
          onClick={() => onUpdateConfig({ showGrid: !config.showGrid })}
          title="Toggle Ground Plane Grid"
        >
          <Grid size={13} />
        </button>
      </div>

      <div className="toolbar-separator" />

      {/* Background Selector */}
      <div className="toolbar-group">
        <select
          className="toolbar-select"
          value={config.background}
          onChange={(e) => onUpdateConfig({ background: e.target.value as BackgroundTheme })}
          title="Select Viewport Background Theme"
        >
          {bgThemes.map((t) => (
            <option key={t.key} value={t.key}>
              {t.label}
            </option>
          ))}
        </select>
      </div>

      {/* Right-aligned QA & Export Actions */}
      <div className="toolbar-right">
        <button className="toolbar-btn-accent" onClick={onOpenQualityModal} title="Review Gate A-E Validation Report">
          <FileCheck2 size={13} />
          <span>QA Report</span>
        </button>

        <button className="toolbar-btn" onClick={handleDownload} title="Download Active 3D Geometry File">
          <Download size={13} />
          <span>Download</span>
        </button>
      </div>
    </div>
  );
};

