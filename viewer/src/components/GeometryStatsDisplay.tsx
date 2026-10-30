import React from 'react';
import { Layers, Box } from 'lucide-react';
import type { ActiveGeometryStats, BoundingBoxInfo } from '../types';

interface GeometryStatsDisplayProps {
  stats: ActiveGeometryStats | null;
  boundingBox: BoundingBoxInfo | null;
}

export const GeometryStatsDisplay: React.FC<GeometryStatsDisplayProps> = ({
  stats,
  boundingBox,
}) => {
  const formatCount = (val: number | null | undefined): string => {
    if (val === null || val === undefined || isNaN(val)) return '0';
    return val.toLocaleString();
  };

  return (
    <div className="geometry-stats-floating">
      <div className="stats-header">
        <Layers size={12} className="text-cyan" />
        <span>Live Geometry Telemetry</span>
      </div>

      <div className="stats-grid">
        <div className="stat-entry">
          <span className="stat-label">Source Asset</span>
          <span className="stat-val mono text-cyan truncate" title={stats?.sourceFileName || ''}>
            {stats?.sourceFileName || 'sample_room.obj'}
          </span>
        </div>

        <div className="stat-entry">
          <span className="stat-label">Unique Vertices</span>
          <span className="stat-val">{formatCount(stats?.uniqueVertices)}</span>
        </div>

        <div className="stat-entry">
          <span className="stat-label">Triangles / Faces</span>
          <span className="stat-val">{formatCount(stats?.triangles)}</span>
        </div>

        <div className="stat-entry">
          <span className="stat-label">Format Buffer</span>
          <span className="stat-val">
            <Box size={10} style={{ display: 'inline', marginRight: 3, verticalAlign: -1 }} />
            {stats?.format === 'OBJ'
              ? 'Non-Indexed OBJ'
              : stats?.format === 'PLY'
              ? 'Indexed PLY'
              : 'Mesh'}
          </span>
        </div>

        {boundingBox && (
          <div className="stat-entry full-width">
            <span className="stat-label">Bounding Extents (X &times; Y &times; Z)</span>
            <span className="stat-val mono">
              {boundingBox.size[0].toFixed(2)}m &times; {boundingBox.size[1].toFixed(2)}m &times;{' '}
              {boundingBox.size[2].toFixed(2)}m
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

