import React, { useEffect, useState } from 'react';
import {
  CheckCircle,
  AlertTriangle,
  Clock,
  RefreshCw,
  Eye,
  Footprints,
  Compass,
  FileVideo,
  Layers,
} from 'lucide-react';
import type { RecentModelItem } from '../types';

interface RecentModelsGalleryProps {
  onSelectModel: (model: RecentModelItem) => void;
  activeModelId?: string;
}

export const RecentModelsGallery: React.FC<RecentModelsGalleryProps> = ({
  onSelectModel,
  activeModelId,
}) => {
  const [models, setModels] = useState<RecentModelItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/v1/recent-models');
      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }
      const data: RecentModelItem[] = await res.json();
      setModels(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch recent models');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const formatDate = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="gallery-container">
      {/* Gallery Header */}
      <div className="gallery-header-section">
        <div className="gallery-title-row">
          <div className="title-left">
            <h2 className="gallery-title">Recent 3D Models</h2>
            <p className="gallery-subtitle">
              Verified digital twin reconstructions created by the F.O.R.G.E. pipeline.
              Displaying the five most recent qualifying models sorted newest-first.
            </p>
          </div>
          <button
            className="gallery-refresh-btn"
            onClick={fetchModels}
            disabled={loading}
            title="Refresh qualifying models list"
          >
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            <span>Refresh</span>
          </button>
        </div>

        {/* Qualification Criteria Disclosure */}
        <div className="gallery-criteria-bar">
          <div className="criteria-item">
            <CheckCircle size={14} className="text-emerald" />
            <span>Genuine completion checks passed</span>
          </div>
          <div className="criteria-item">
            <CheckCircle size={14} className="text-emerald" />
            <span>Zero mock or sample fixtures</span>
          </div>
          <div className="criteria-item">
            <CheckCircle size={14} className="text-emerald" />
            <span>Actual GLB binary verified on disk</span>
          </div>
          <div className="criteria-item">
            <Clock size={14} className="text-cyan" />
            <span>Sorted newest first by completion timestamp</span>
          </div>
        </div>
      </div>

      {/* Loading & Error States */}
      {loading && models.length === 0 && (
        <div className="gallery-loading-state">
          <RefreshCw size={32} className="spin text-cyan" />
          <p>Auditing and discovering recent genuine 3D reconstructions...</p>
        </div>
      )}

      {error && (
        <div className="gallery-error-banner">
          <AlertTriangle size={18} />
          <span>Error loading recent models: {error}</span>
          <button onClick={fetchModels}>Retry</button>
        </div>
      )}

      {!loading && models.length === 0 && !error && (
        <div className="gallery-empty-state">
          <Layers size={48} className="text-muted" />
          <h3>No Qualifying Reconstructions Found</h3>
          <p>
            Reconstructions will appear here automatically when they complete through the actual
            pipeline and pass quality checks.
          </p>
        </div>
      )}

      {/* Models Grid */}
      <div className="gallery-grid">
        {models.map((model) => {
          const isSelected = activeModelId === model.id;
          const isReady = model.reconstruction_status === 'WALKTHROUGH_READY';
          const isPartial = model.is_partial || model.reconstruction_status === 'PARTIAL_MODEL';

          return (
            <div
              key={model.id}
              className={`model-card ${isSelected ? 'selected' : ''} ${isPartial ? 'partial' : ''}`}
            >
              {/* Card Preview / Thumbnail */}
              <div className="card-media-wrapper">
                <img
                  src={model.thumbnail_url}
                  alt={model.scene_name}
                  className="card-thumbnail"
                  onError={(e) => {
                    // Fallback to SVG placeholder on error
                    (e.target as HTMLElement).style.display = 'none';
                  }}
                />
                <div className="card-badge-row">
                  {isReady ? (
                    <span className="badge-status ready">
                      <Footprints size={12} />
                      <span>Walkthrough Ready</span>
                    </span>
                  ) : isPartial ? (
                    <span className="badge-status partial">
                      <Compass size={12} />
                      <span>Partial Model &bull; Orbit Inspection</span>
                    </span>
                  ) : (
                    <span className="badge-status verified">
                      <CheckCircle size={12} />
                      <span>Verified Asset</span>
                    </span>
                  )}
                  <span className="badge-size">{model.file_size_mb} MB</span>
                </div>
              </div>

              {/* Card Body */}
              <div className="card-body">
                <h3 className="card-scene-title" title={model.scene_name}>
                  {model.scene_name}
                </h3>

                <div className="card-video-name" title={model.source_video_name}>
                  <FileVideo size={13} className="text-muted flex-shrink-0" />
                  <span className="truncate">{model.source_video_name}</span>
                </div>

                <div className="card-metadata-grid">
                  <div className="meta-cell">
                    <span className="meta-label">Completed</span>
                    <span className="meta-value">{formatDate(model.completion_timestamp)}</span>
                  </div>
                  <div className="meta-cell">
                    <span className="meta-label">Geometry</span>
                    <span className="meta-value">
                      {model.triangle_count ? `${Math.round(model.triangle_count / 1000)}k tris` : 'Dense mesh'}
                    </span>
                  </div>
                  <div className="meta-cell">
                    <span className="meta-label">Floor Continuity</span>
                    <span className="meta-value">{model.floor_continuity_pct}%</span>
                  </div>
                  <div className="meta-cell">
                    <span className="meta-label">Clearance</span>
                    <span className="meta-value">{model.obstacle_clearance_pct}%</span>
                  </div>
                </div>

                {isPartial && (
                  <div className="partial-notice">
                    <AlertTriangle size={12} />
                    <span>Navigation restricted to verified sections; orbit inspection mode available.</span>
                  </div>
                )}
              </div>

              {/* Card Action */}
              <div className="card-footer">
                <button
                  className={`card-select-btn ${isSelected ? 'active' : ''}`}
                  onClick={() => onSelectModel(model)}
                  title={`Open ${model.scene_name} in 3D Viewer`}
                >
                  <Eye size={14} />
                  <span>{isSelected ? 'Currently Viewing' : 'Explore in 3D Viewer'}</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
