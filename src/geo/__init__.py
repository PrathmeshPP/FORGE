"""F.O.R.G.E Geospatial Alignment & Telemetry Module."""

from src.geo.telemetry import (
    GPSTelemetryRecord,
    extract_dji_telemetry,
    match_telemetry_to_frame_timestamps,
    geodetic_to_enu,
    enu_to_geodetic
)
from src.geo.umeyama_ransac import (
    AlignmentResult,
    umeyama_alignment,
    umeyama_ransac_alignment,
    apply_similarity_transform,
    transform_camera_poses,
    transform_mesh
)

from src.geo.quality_gates import (
    QualityCheck,
    QualityGateReport,
    evaluate_reconstruction_quality_gate
)

from src.geo.dsm_exporter import (
    CoordinateReferenceMetadata,
    DSMConfig,
    DSMExportResult,
    DSMExporter,
    UnreferencedGeometryError,
    GeographicCRSError,
    IncompatibleUnitsError,
    UnvalidatedAlignmentError,
    InvalidRasterConfigurationError,
    derive_utm_crs_from_latlon,
    reproject_geographic_to_utm,
    rasterize_point_cloud_dsm
)

__all__ = [
    "GPSTelemetryRecord",
    "extract_dji_telemetry",
    "match_telemetry_to_frame_timestamps",
    "geodetic_to_enu",
    "enu_to_geodetic",
    "AlignmentResult",
    "umeyama_alignment",
    "umeyama_ransac_alignment",
    "apply_similarity_transform",
    "transform_camera_poses",
    "transform_mesh",
    "QualityCheck",
    "QualityGateReport",
    "evaluate_reconstruction_quality_gate",
    "CoordinateReferenceMetadata",
    "DSMConfig",
    "DSMExportResult",
    "DSMExporter",
    "UnreferencedGeometryError",
    "GeographicCRSError",
    "IncompatibleUnitsError",
    "UnvalidatedAlignmentError",
    "InvalidRasterConfigurationError",
    "derive_utm_crs_from_latlon",
    "reproject_geographic_to_utm",
    "rasterize_point_cloud_dsm",
]
