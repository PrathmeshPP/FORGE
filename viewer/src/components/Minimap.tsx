import React, { useEffect, useRef } from 'react';

interface MinimapProps {
  cameraPos: { x: number; y: number; z: number };
  cameraYaw: number; // in radians
  bounds: { minX: number; maxX: number; minZ: number; maxZ: number } | null;
  footprintPoints?: { x: number; z: number }[];
}

export const Minimap: React.FC<MinimapProps> = ({
  cameraPos,
  cameraYaw,
  bounds,
  footprintPoints = [],
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Background circle/radar
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = width / 2 - 6;

    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.stroke();
    ctx.clip(); // Keep drawing inside radar circle

    // Draw concentric radar rings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    [0.35, 0.7, 0.95].forEach((factor) => {
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * factor, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - radius);
    ctx.lineTo(centerX, centerY + radius);
    ctx.moveTo(centerX - radius, centerY);
    ctx.lineTo(centerX + radius, centerY);
    ctx.stroke();

    // Map bounds calculation
    const margin = 0.5;
    let bMinX = bounds ? bounds.minX - margin : -3;
    let bMaxX = bounds ? bounds.maxX + margin : 3;
    let bMinZ = bounds ? bounds.minZ - margin : -3;
    let bMaxZ = bounds ? bounds.maxZ + margin : 3;

    const spanX = Math.max(0.1, bMaxX - bMinX);
    const spanZ = Math.max(0.1, bMaxZ - bMinZ);
    const maxSpan = Math.max(spanX, spanZ);

    const scale = (radius * 1.5) / maxSpan;
    const midX = (bMinX + bMaxX) / 2;
    const midZ = (bMinZ + bMaxZ) / 2;

    const worldToCanvas = (wx: number, wz: number) => {
      const cx = centerX + (wx - midX) * scale;
      const cy = centerY + (wz - midZ) * scale;
      return { cx, cy };
    };

    // Draw room footprint / bounds
    if (footprintPoints.length > 2) {
      ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const p0 = worldToCanvas(footprintPoints[0].x, footprintPoints[0].z);
      ctx.moveTo(p0.cx, p0.cy);
      for (let i = 1; i < footprintPoints.length; i++) {
        const pt = worldToCanvas(footprintPoints[i].x, footprintPoints[i].z);
        ctx.lineTo(pt.cx, pt.cy);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else if (bounds) {
      // Draw rectangular bounds
      const tl = worldToCanvas(bounds.minX, bounds.minZ);
      const br = worldToCanvas(bounds.maxX, bounds.maxZ);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.rect(tl.cx, tl.cy, br.cx - tl.cx, br.cy - tl.cy);
      ctx.fill();
      ctx.stroke();
    }

    // Draw player position and heading
    const playerC = worldToCanvas(cameraPos.x, cameraPos.z);

    // Heading cone (Field of View indicator)
    const coneLen = 22;
    const fovHalf = Math.PI / 6; // 30 deg
    ctx.save();
    ctx.translate(playerC.cx, playerC.cy);
    ctx.rotate(cameraYaw);

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, coneLen, -Math.PI / 2 - fovHalf, -Math.PI / 2 + fovHalf);
    ctx.closePath();
    ctx.fillStyle = 'rgba(16, 185, 129, 0.35)';
    ctx.fill();

    // Direction line
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, -coneLen);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Player dot
    ctx.beginPath();
    ctx.arc(0, 0, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();

    ctx.restore(); // Restore outer clip

    // North indicator
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('N', centerX, 15);
  }, [cameraPos, cameraYaw, bounds, footprintPoints]);

  return (
    <div
      style={{
        position: 'absolute',
        top: '16px',
        right: '16px',
        width: '180px',
        height: '180px',
        zIndex: 20,
        pointerEvents: 'none',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <canvas
        ref={canvasRef}
        width={180}
        height={180}
        style={{
          width: '180px',
          height: '180px',
          filter: 'drop-shadow(0 6px 20px rgba(0, 0, 0, 0.6))',
        }}
      />
      <div
        style={{
          marginTop: '-18px',
          padding: '2px 8px',
          background: 'rgba(15, 23, 42, 0.9)',
          border: '1px solid rgba(56, 189, 248, 0.4)',
          borderRadius: '4px',
          color: '#94a3b8',
          fontSize: '10px',
          fontFamily: 'monospace',
          letterSpacing: '0.5px',
        }}
      >
        MAP • ({cameraPos.x.toFixed(1)}, {cameraPos.z.toFixed(1)})m
      </div>
    </div>
  );
};
