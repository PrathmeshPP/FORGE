import React, { useRef, useState } from 'react';
import {
  UploadCloud,
  FileVideo,
  Shield,
  HardDrive,
  Info,
  CheckCircle,
  Eye,
} from 'lucide-react';
import type { PreflightMetadata } from '../types';

interface UploadViewProps {
  onVideoSelected: (file: File, preflight: PreflightMetadata, keepRaw: boolean) => void;
  onLoadSampleModel: () => void;
  maxUploadBytes: number;
  acceptedExtensions: string[];
}

export const UploadView: React.FC<UploadViewProps> = ({
  onVideoSelected,
  onLoadSampleModel,
  maxUploadBytes,
  acceptedExtensions,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [keepRawInput, setKeepRawInput] = useState(false);
  const [analyzingFile, setAnalyzingFile] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const formatBytes = (bytes: number) => {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    setErrorMessage(null);

    // 1. Basic Extension Check
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedExtensions.includes(ext)) {
      setErrorMessage(
        `Unsupported video container "${ext}". Only ${acceptedExtensions.join(', ')} files are supported for indoor reconstruction.`
      );
      return;
    }

    // 2. Size Limit Check
    if (file.size > maxUploadBytes) {
      setErrorMessage(
        `File size (${(file.size / (1024 * 1024 * 1024)).toFixed(2)} GB) exceeds the maximum configured limit of ${formatBytes(
          maxUploadBytes
        )}.`
      );
      return;
    }

    // 3. Client-side Video Probe
    setAnalyzingFile(true);
    try {
      const metadata = await probeVideoFile(file);
      setAnalyzingFile(false);
      onVideoSelected(file, metadata, keepRawInput);
    } catch (err: any) {
      setAnalyzingFile(false);
      setErrorMessage(err.message || 'Failed to parse video metadata. File may be corrupt or unreadable.');
    }
  };

  const probeVideoFile = (file: File): Promise<PreflightMetadata> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      const objectUrl = URL.createObjectURL(file);

      const timeout = setTimeout(() => {
        URL.revokeObjectURL(objectUrl);
        reject(new Error('Video probing timed out. Container headers could not be decoded.'));
      }, 8000);

      video.onloadedmetadata = () => {
        clearTimeout(timeout);
        URL.revokeObjectURL(objectUrl);

        const duration = video.duration || 0;
        const width = video.videoWidth || 0;
        const height = video.videoHeight || 0;
        const sizeMb = file.size / (1024 * 1024);

        const warnings: string[] = [];
        const rejectionReasons: string[] = [];

        if (duration > 900) {
          rejectionReasons.push('Video duration exceeds 15 minutes limit (900 seconds).');
        } else if (duration < 3) {
          rejectionReasons.push('Video duration is under 3 seconds; insufficient coverage for 3D reconstruction.');
        }

        if (width < 640 || height < 480) {
          rejectionReasons.push(`Resolution ${width}x${height} is too low for reliable geometric feature matching.`);
        } else if (width < 1280 || height < 720) {
          warnings.push('Sub-720p resolution detected. Geometric reconstruction quality may be degraded.');
        }

        const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
        const divisor = gcd(width, height);
        const aspect = `${width / divisor}:${height / divisor}`;

        const status = rejectionReasons.length > 0 ? 'rejected' : warnings.length > 0 ? 'warning' : 'passed';

        resolve({
          filename: file.name,
          file_size_bytes: file.size,
          file_size_mb: sizeMb,
          mime_type: file.type || 'video/mp4',
          container_valid: true,
          duration_seconds: duration,
          width,
          height,
          aspect_ratio: aspect,
          status,
          rejection_reasons: rejectionReasons,
          warnings,
        });
      };

      video.onerror = () => {
        clearTimeout(timeout);
        URL.revokeObjectURL(objectUrl);
        reject(new Error('Browser video decoder rejected this file. Ensure it is a valid H.264/HEVC MP4 or MOV.'));
      };

      video.src = objectUrl;
    });
  };

  return (
    <div className="upload-view-container">
      {/* Privacy Notice Banner */}
      <div className="privacy-banner">
        <div className="banner-icon-col">
          <Shield size={18} className="text-emerald" />
        </div>
        <div className="banner-content">
          <span className="banner-title">Private Local Processing Guarantee</span>
          <p className="banner-desc">
            All spatial reconstruction runs entirely on this Windows workstation (NVIDIA GeForce RTX 2070 SUPER / 8 GB VRAM).
            Uploaded media is never sent to AWS S3, cloud APIs, or third parties. Access is restricted to your private
            Tailscale network.
          </p>
        </div>
      </div>

      <div className="upload-card">
        <div className="card-header-clean">
          <h2>Submit Indoor Smartphone Video</h2>
          <p className="subtitle">
            Walk slowly through the indoor room or corridor keeping the camera pointed at stable surfaces and furniture.
          </p>
        </div>

        {/* Drag & Drop Zone */}
        <div
          className={`dropzone ${isDragging ? 'dragging' : ''} ${analyzingFile ? 'analyzing' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              fileInputRef.current?.click();
            }
          }}
        >
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            accept={acceptedExtensions.join(',')}
            onChange={(e) => handleFiles(e.target.files)}
          />

          <div className="dropzone-inner">
            <div className="dropzone-icon-circle">
              {analyzingFile ? (
                <FileVideo size={28} className="pulse-icon text-cyan" />
              ) : (
                <UploadCloud size={28} className="text-cyan" />
              )}
            </div>

            <div className="dropzone-text">
              <span className="drop-title">
                {analyzingFile ? 'Analyzing video container & stream...' : 'Choose indoor video or drag & drop here'}
              </span>
              <span className="drop-sub">
                Supported containers: <strong>{acceptedExtensions.join(', ')}</strong> &bull; Max file size:{' '}
                <strong>{formatBytes(maxUploadBytes)}</strong>
              </span>
            </div>

            <button
              type="button"
              className="btn-primary"
              disabled={analyzingFile}
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Select File from Workstation
            </button>
          </div>
        </div>

        {errorMessage && (
          <div className="error-alert">
            <Info size={16} />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Retention Policy Checkbox */}
        <div className="upload-options">
          <label className="checkbox-label" title="Default behavior deletes source video immediately after terminal job state">
            <input
              type="checkbox"
              checked={keepRawInput}
              onChange={(e) => setKeepRawInput(e.target.checked)}
            />
            <span className="checkbox-text">
              Keep original video file on disk for 24 hours (reproducibility cache)
            </span>
          </label>
          <span className="option-hint">
            By default, raw videos are deleted immediately upon completion to conserve local disk space and protect privacy.
          </span>
        </div>

        {/* Demo / Sample Model Shortcut */}
        <div className="sample-model-section">
          <div className="sample-divider">
            <span>OR INSPECT TEST FIXTURE</span>
          </div>
          <div className="sample-box">
            <div className="sample-meta">
              <span className="sample-title">
                <CheckCircle size={14} className="text-emerald" style={{ display: 'inline', marginRight: 4 }} />
                Instant 3D Viewer Inspection Mode
              </span>
              <p className="sample-desc">
                Load the verified indoor office sample model (Textured OBJ / Stanford PLY) into the Three.js viewport without
                uploading video or running reconstruction.
              </p>
            </div>
            <button type="button" className="btn-secondary" onClick={onLoadSampleModel}>
              <Eye size={14} />
              <span>Open Sample 3D Model</span>
            </button>
          </div>
        </div>
      </div>

      {/* Technical Best Practices Card */}
      <div className="guidelines-card">
        <h3>
          <HardDrive size={15} style={{ display: 'inline', marginRight: 6, verticalAlign: -2 }} />
          Capture Recommendations for 8 GB VRAM Workstation
        </h3>
        <ul className="guidelines-list">
          <li>
            <strong>Movement:</strong> Walk continuously along smooth paths. Avoid rapid panning or quick rotational jerky turns.
          </li>
          <li>
            <strong>Lighting:</strong> Ensure even indoor illumination. Avoid direct pointing into intensely bright windows.
          </li>
          <li>
            <strong>Coverage:</strong> Keep furniture, doorways, and floor boundaries in frame across multiple vantage points.
          </li>
          <li>
            <strong>Duration:</strong> 15 to 90 seconds recommended for typical single rooms to maintain high frame density.
          </li>
        </ul>
      </div>
    </div>
  );
};

