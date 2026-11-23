import React, { useRef } from 'react';

export interface ModelOption {
  id: string;
  name: string;
  url: string;
  sizeMb?: number;
  source?: 'server' | 'local';
}

interface TopBarProps {
  models: ModelOption[];
  selectedModelId: string;
  onSelectModel: (model: ModelOption) => void;
  onUploadLocalFile: (file: File) => void;
  onResetCamera: () => void;
  isLoading: boolean;
  loadingProgress: number;
}

export const TopBar: React.FC<TopBarProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  onUploadLocalFile,
  onResetCamera,
  isLoading,
  loadingProgress,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadLocalFile(file);
    }
  };

  return (
    <header
      style={{
        position: 'absolute',
        top: '16px',
        left: '16px',
        zIndex: 20,
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '8px 16px',
        background: 'rgba(15, 23, 42, 0.90)',
        border: '1px solid rgba(255, 255, 255, 0.12)',
        borderRadius: '10px',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
        backdropFilter: 'blur(10px)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div
          style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            background: isLoading ? '#f59e0b' : '#10b981',
            boxShadow: isLoading
              ? '0 0 8px #f59e0b'
              : '0 0 8px #10b981',
          }}
        />
        <span
          style={{
            fontSize: '13px',
            fontWeight: 700,
            color: '#f8fafc',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            letterSpacing: '0.5px',
          }}
        >
          F.O.R.G.E.
        </span>
      </div>

      <div style={{ width: '1px', height: '18px', background: 'rgba(255, 255, 255, 0.15)' }} />

      {/* Model Selection Dropdown */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <label
          htmlFor="model-select"
          style={{
            fontSize: '12px',
            color: '#94a3b8',
            fontFamily: 'system-ui, sans-serif',
          }}
        >
          Model:
        </label>
        <select
          id="model-select"
          value={selectedModelId}
          onChange={(e) => {
            const found = models.find((m) => m.id === e.target.value);
            if (found) onSelectModel(found);
          }}
          disabled={isLoading}
          style={{
            background: '#1e293b',
            color: '#e2e8f0',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: '6px',
            padding: '5px 10px',
            fontSize: '12px',
            fontFamily: 'system-ui, sans-serif',
            cursor: 'pointer',
            maxWidth: '260px',
            outline: 'none',
          }}
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} {m.sizeMb ? `(${m.sizeMb} MB)` : ''}
            </option>
          ))}
        </select>
      </div>

      {/* File Upload Input & Button */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".glb,.gltf"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={isLoading}
        title="Open and navigate any 3D model (.glb) from your PC"
        style={{
          background: 'rgba(56, 189, 248, 0.15)',
          color: '#38bdf8',
          border: '1px solid rgba(56, 189, 248, 0.35)',
          borderRadius: '6px',
          padding: '5px 12px',
          fontSize: '12px',
          fontWeight: 600,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          transition: 'all 0.15s ease',
        }}
      >
        <span>📁</span> Open .GLB from PC
      </button>

      {/* Reset Camera View */}
      <button
        type="button"
        onClick={onResetCamera}
        disabled={isLoading}
        title="Reset camera to initial viewpoint"
        style={{
          background: 'rgba(255, 255, 255, 0.05)',
          color: '#cbd5e1',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: '6px',
          padding: '5px 10px',
          fontSize: '12px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}
      >
        <span>🔄</span> Reset
      </button>

      {/* Loading Progress */}
      {isLoading && (
        <span style={{ fontSize: '12px', color: '#f59e0b', fontFamily: 'monospace' }}>
          Loading {loadingProgress > 0 ? `${loadingProgress}%` : '...'}
        </span>
      )}
    </header>
  );
};
