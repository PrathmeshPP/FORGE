import React from 'react';
import {
  Grid,
  Compass,
  RotateCcw,
  Eye,
  FileCheck2,
  Download,
  Sun,
  Footprints,
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
  isWalkthroughMode?: boolean;
  onToggleWalkthrough?: () => void;
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
  isWalkthroughMode,
  onToggleWalkthrough,
}) => {
  const bgThemes: { key: BackgroundTheme; label: string }[] = [
    { key: 'slate', label: 'Dark Slate' },
    { key: 'charcoal', label: 'Charcoal' },
    { key: 'black', label: 'Pitch Black' },
    { key: 'light', label: 'Light Studio' },
  ];

  const handleDownload = () => {
    const rep = scene.representations[currentRepresentation];
    const url = rep?.glb_url || rep?.obj_url || rep?.ply_url;
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
                {repKey === 'glb'
                  ? 'GLB Photographic'
                  : repKey === 'textured_obj'
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

      {/* Brightness Control */}
      <div className="toolbar-group brightness-control-group" title="Adjust Viewport Brightness (0.5x - 2.0x)">
        <Sun size={13} className="text-amber" />
        <input
          type="range"
          min="0.5"
          max="2.0"
          step="0.05"
          value={config.brightness ?? 1.0}
          onChange={(e) => onUpdateConfig({ brightness: parseFloat(e.target.value) })}
          className="brightness-slider"
          title={`Brightness: ${Math.round((config.brightness ?? 1.0) * 100)}% (${(config.brightness ?? 1.0).toFixed(2)}x)`}
        />
        <span className="brightness-value-label">
          {Math.round((config.brightness ?? 1.0) * 100)}%
        </span>
        <button
          className="toolbar-btn icon-only reset-brightness-btn"
          onClick={() => onUpdateConfig({ brightness: 1.0 })}
          title="Reset Brightness to 1.0x (100%)"
          disabled={(config.brightness ?? 1.0) === 1.0}
        >
          <RotateCcw size={11} />
        </button>
      </div>

      <div className="toolbar-separator" />

      {/* Walkthrough Mode Toggle */}
      {onToggleWalkthrough && (
        <button
          className={`toolbar-btn walkthrough-btn ${isWalkthroughMode ? 'active' : ''}`}
          onClick={onToggleWalkthrough}
          title={isWalkthroughMode ? 'Exit Walkthrough Mode (Return to Orbit)' : 'Enter First-Person Free-Roam Mode (WASD + UP)'}
        >
          <Footprints size={13} />
          <span>{isWalkthroughMode ? 'Exit Walk' : 'Walkthrough'}</span>
        </button>
      )}

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

