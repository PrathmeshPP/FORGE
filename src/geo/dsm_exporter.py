"""F.O.R.G.E — Digital Surface Model (DSM) GeoTIFF Exporter.

Provides rigorous, validated generation of Digital Surface Model (DSM) rasters from 3D point
clouds with explicit geodetic Coordinate Reference System (CRS) verification, metric unit checks,
georeferencing assurance levels, vertical datum limitation documentation, and NoData management.

CRITICAL METHODOLOGICAL DISTINCTION:
This module exports a Digital Surface Model (DSM)—representing the upper reflective surface
envelope (ground, buildings, infrastructure, and vegetation canopies). It does NOT perform
progressive morphological ground filtering or bare-earth separation, and is therefore NEVER
labeled as a bare-earth Digital Elevation Model (DEM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import pyproj
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS


class UnreferencedGeometryError(ValueError):
    """Raised when attempting georeferenced GeoTIFF export on unreferenced/model-relative coordinates."""
    pass


class GeographicCRSError(ValueError):
    """Raised when a geographic (angular degree) CRS is used with metric GSD without reprojection."""
    pass


class IncompatibleUnitsError(UnreferencedGeometryError):
    """Raised when horizontal or vertical units are incompatible with metric rasterization."""
    pass


class UnvalidatedAlignmentError(ValueError):
    """Raised when georeferencing metadata lacks sufficient evidence of usable alignment quality."""
    pass


class InvalidRasterConfigurationError(ValueError):
    """Raised when raster grid bounds, GSD, or dimensions are degenerate."""
    pass


def derive_utm_crs_from_latlon(lat: float, lon: float) -> int:
    """
    Computes standard WGS84 Universal Transverse Mercator (UTM) EPSG code.

    Args:
        lat: Latitude in decimal degrees [-90.0, 90.0].
        lon: Longitude in decimal degrees [-180.0, 180.0].

    Returns:
        EPSG integer code (32601-32660 for North, 32701-32760 for South).
    """
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"Geodetic coordinates out of bounds: lat={lat}, lon={lon}")

    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    if zone > 60:
        zone = 60

    if lat >= 0.0:
        return 32600 + zone
    else:
        return 32700 + zone


def reproject_geographic_to_utm(
    points_lon_lat_alt: np.ndarray,
    source_crs: Union[str, int] = "EPSG:4326",
    target_utm_epsg: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Reprojects 3D coordinates from a Geographic CRS (lon, lat, alt in degrees/meters)
    to a projected WGS84 UTM metric coordinate system (easting, northing, alt in meters).

    Args:
        points_lon_lat_alt: (N, 3) array of [Longitude, Latitude, Altitude].
        source_crs: Source geographic CRS string or EPSG code (default: EPSG:4326).
        target_utm_epsg: Optional target UTM EPSG. If None, derived from median coordinate.

    Returns:
        Tuple (projected_points_easting_northing_alt, utm_epsg_int).
    """
    pts = np.asarray(points_lon_lat_alt, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected (N, 3) point array, got shape {pts.shape}")

    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.any(finite_mask):
        raise ValueError("Cannot reproject point cloud: no finite coordinates.")

    med_lon = float(np.median(pts[finite_mask, 0]))
    med_lat = float(np.median(pts[finite_mask, 1]))

    if target_utm_epsg is None:
        target_utm_epsg = derive_utm_crs_from_latlon(med_lat, med_lon)

    src_crs_str = f"EPSG:{source_crs}" if isinstance(source_crs, int) else str(source_crs)
    transformer = pyproj.Transformer.from_crs(src_crs_str, f"EPSG:{target_utm_epsg}", always_xy=True)
    easting, northing = transformer.transform(pts[:, 0], pts[:, 1])
    projected = np.column_stack([easting, northing, pts[:, 2]])

    return projected, target_utm_epsg


@dataclass
class CoordinateReferenceMetadata:
    """
    Encapsulates validated georeferencing, coordinate frame, and quality metadata.

    A valid Sim(3) transform or GPS log alone is NOT treated as proof of accurate georeferencing;
    explicit coordinate reference information and quality gates must be satisfied.
    """
    is_georeferenced: bool
    crs_epsg: Optional[int] = None
    crs_wkt: Optional[str] = None
    horizontal_units: str = "meters"
    vertical_units: str = "meters"
    anchor_geodetic: Optional[Dict[str, float]] = None
    sim3_scale: Optional[float] = None
    alignment_rmse_m: Optional[float] = None
    alignment_inlier_ratio: Optional[float] = None
    provenance_note: str = ""
    vertical_datum: str = "WGS84_ELLIPSOID_OR_TAKEOFF_RELATIVE"
    vertical_datum_note: str = (
        "Elevations are ellipsoidal or relative barometric heights; "
        "not converted to orthometric geoid (e.g. EGM96/MSL) without surveyed GCPs."
    )

    def validate_for_georeferenced_export(
        self,
        require_accuracy_evidence: bool = True,
        min_inlier_ratio: float = 0.50,
        max_rmse_m: float = 10.0,
    ) -> Tuple[bool, str, str]:
        """
        Validates whether metadata supports a defensible, georeferenced GIS export.

        Args:
            require_accuracy_evidence: If True, requires valid alignment_rmse_m and
                                       alignment_inlier_ratio to certify ACCURACY_VALIDATED.
                                       If False, allows EXPLORATORY_ESTIMATE outputs.
            min_inlier_ratio: Minimum acceptable RANSAC consensus inlier ratio.
            max_rmse_m: Maximum acceptable residual RMSE in meters.

        Returns:
            Tuple (is_valid, assurance_level, reason_or_status).
            assurance_level is one of: "ACCURACY_VALIDATED", "EXPLORATORY", "GEOGRAPHIC",
                                       "INCOMPATIBLE_UNITS", "UNREFERENCED".
        """
        if not self.is_georeferenced:
            return False, "UNREFERENCED", "Geometry is explicitly marked as non-georeferenced / model-relative."

        if self.crs_epsg is None and not self.crs_wkt:
            return False, "UNREFERENCED", "Missing coordinate reference system (neither crs_epsg nor crs_wkt provided)."

        # 1. Inspect CRS angular vs metric nature
        try:
            if self.crs_epsg is not None:
                parsed_crs = CRS.from_epsg(self.crs_epsg)
            else:
                parsed_crs = CRS.from_wkt(self.crs_wkt)
        except Exception as e:
            return False, "UNREFERENCED", f"Failed to parse CRS: {str(e)}"

        if parsed_crs.is_geographic or self.horizontal_units.lower() in ["degrees", "degree", "deg"]:
            return False, "GEOGRAPHIC", (
                f"CRS (EPSG:{self.crs_epsg or 'WKT'}) is geographic (coordinates in angular degrees). "
                "Metric GSD rasterization requires a projected metric CRS (e.g. UTM) where horizontal coordinates "
                "are expressed in meters. Reproject coordinates before rasterization or enable auto_reproject_geographic."
            )

        # 2. Inspect units
        h_unit = self.horizontal_units.lower()
        if h_unit not in ["meters", "metre", "m"]:
            return False, "INCOMPATIBLE_UNITS", (
                f"Incompatible horizontal units '{self.horizontal_units}'. Metric GSD requires 'meters'."
            )

        v_unit = self.vertical_units.lower()
        if v_unit not in ["meters", "metre", "m"]:
            return False, "INCOMPATIBLE_UNITS", (
                f"Incompatible vertical units '{self.vertical_units}'. Expected 'meters'."
            )

        # 3. Verify scale factor if provided
        if self.sim3_scale is not None:
            if not np.isfinite(self.sim3_scale) or self.sim3_scale <= 0.0:
                return False, "UNREFERENCED", f"Degenerate or negative similarity scale factor: {self.sim3_scale}."

        # 4. Check alignment quality gates
        has_rmse = self.alignment_rmse_m is not None and np.isfinite(self.alignment_rmse_m)
        has_inliers = self.alignment_inlier_ratio is not None and np.isfinite(self.alignment_inlier_ratio)

        if require_accuracy_evidence:
            if not has_rmse or not has_inliers:
                return False, "EXPLORATORY", (
                    "Missing alignment quality evidence: both alignment_rmse_m and alignment_inlier_ratio "
                    "must be provided to certify positional accuracy for standard georeferenced export. "
                    "To export an uncertified exploratory raster, set allow_exploratory_georeference=True."
                )

            if self.alignment_rmse_m > max_rmse_m:
                return False, "EXPLORATORY", (
                    f"Geospatial alignment residual RMSE ({self.alignment_rmse_m:.2f}m) exceeds "
                    f"acceptable quality ceiling ({max_rmse_m:.2f}m)."
                )

            if self.alignment_inlier_ratio < min_inlier_ratio:
                return False, "EXPLORATORY", (
                    f"Geospatial alignment inlier ratio ({self.alignment_inlier_ratio:.2%}) is below "
                    f"acceptable confidence floor ({min_inlier_ratio:.2%})."
                )

            return True, "TELEMETRY_ALIGNED_CONSISTENT", "Internal drone telemetry alignment consistency satisfied (RMSE <= 10m, inliers >= 50%; not survey-grade)."

        # Exploratory mode permitted
        if has_rmse and self.alignment_rmse_m > 50.0:
            return False, "EXPLORATORY", f"Alignment RMSE ({self.alignment_rmse_m:.2f}m) indicates catastrophic failure."

        if has_rmse and has_inliers and self.alignment_rmse_m <= max_rmse_m and self.alignment_inlier_ratio >= min_inlier_ratio:
            return True, "TELEMETRY_ALIGNED_CONSISTENT", "Internal drone telemetry alignment consistency satisfied (RMSE <= 10m, inliers >= 50%; not survey-grade)."
        else:
            return True, "EXPLORATORY", "Exploratory georeferenced output without certified accuracy."


@dataclass
class DSMConfig:
    """Configuration for Digital Surface Model rasterization."""
    gsd_m: float = 0.25  # Ground Sampling Distance (meters/pixel)
    nodata_value: float = -9999.0
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, min_y, max_x, max_y) in meters
    dtype: str = "float32"
    target_product_type: str = "DSM"  # Strictly Digital Surface Model


@dataclass
class DSMExportResult:
    """Summary of the exported Digital Surface Model raster."""
    success: bool
    output_path: str
    is_georeferenced: bool
    assurance_level: str  # "ACCURACY_VALIDATED", "EXPLORATORY", "UNREFERENCED_LOCAL"
    crs_string: Optional[str]
    raster_width: int
    raster_height: int
    gsd: float
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    elevation_min: float
    elevation_max: float
    valid_cell_count: int
    nodata_cell_count: int
    valid_coverage_pct: float
    metadata: Dict[str, Any]
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def rasterize_point_cloud_dsm(
    points: np.ndarray,
    gsd: float,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    nodata_val: float = -9999.0,
) -> Tuple[np.ndarray, Affine, Tuple[float, float, float, float]]:
    """
    Rasterizes a 3D point cloud into a 2D surface grid taking the maximum elevation per cell.

    Args:
        points: (N, 3) array of [X, Y, Z] point coordinates in metric space.
        gsd: Ground Sampling Distance per pixel in meters.
        bounds: Optional (min_x, min_y, max_x, max_y) in meters. If None, calculated from points.
        nodata_val: Sentinel value for unobserved grid cells.

    Returns:
        Tuple of:
          - 2D numpy array of shape (height, width) with float32 elevations.
          - Affine transform mapping pixel (col, row) to metric coordinate space.
          - Actual bounding box (min_x, min_y, max_x, max_y).
    """
    if gsd <= 0.0 or not np.isfinite(gsd):
        raise InvalidRasterConfigurationError(f"GSD must be positive and finite. Got {gsd}")

    # 1. Filter non-finite coordinates
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected (N, 3) point array, got shape {pts.shape}")

    finite_mask = np.isfinite(pts).all(axis=1)
    pts_valid = pts[finite_mask]

    if len(pts_valid) == 0:
        raise ValueError("Point cloud contains zero valid, finite points.")

    x_vals = pts_valid[:, 0]
    y_vals = pts_valid[:, 1]
    z_vals = pts_valid[:, 2]

    # 2. Determine spatial extent
    if bounds is not None:
        min_x, min_y, max_x, max_y = bounds
        if not (min_x < max_x and min_y < max_y):
            raise InvalidRasterConfigurationError(f"Invalid bounding box: {bounds}")
    else:
        min_x = float(np.min(x_vals))
        max_x = float(np.max(x_vals))
        min_y = float(np.min(y_vals))
        max_y = float(np.max(y_vals))

        # Handle degenerate single-point or collinear clouds
        if max_x == min_x:
            max_x = min_x + gsd
        if max_y == min_y:
            max_y = min_y + gsd

    width = int(math.ceil((max_x - min_x) / gsd))
    height = int(math.ceil((max_y - min_y) / gsd))

    # Guard against zero-area or runaway dimensions
    if width <= 0 or height <= 0:
        raise InvalidRasterConfigurationError(f"Degenerate raster dimensions: {width}x{height}")
    if width > 16384 or height > 16384:
        raise InvalidRasterConfigurationError(
            f"Raster dimensions {width}x{height} exceed memory safety threshold (16384 max)."
        )

    # Adjust max extent to align with integer pixel boundary
    adjusted_max_x = min_x + width * gsd
    adjusted_max_y = min_y + height * gsd
    actual_bounds = (min_x, min_y, adjusted_max_x, adjusted_max_y)

    # Standard North-up top-left origin affine matrix:
    # X_geo = transform.c + (col + 0.5) * transform.a
    # Y_geo = transform.f + (row + 0.5) * transform.e
    transform = Affine(gsd, 0.0, min_x, 0.0, -gsd, adjusted_max_y)

    # 3. Bin points into grid cells with robust boundary handling
    in_bounds = (x_vals >= min_x) & (x_vals <= adjusted_max_x) & (y_vals >= min_y) & (y_vals <= adjusted_max_y)
    if not np.any(in_bounds):
        grid = np.full((height, width), nodata_val, dtype=np.float32)
        return grid, transform, actual_bounds

    x_in = x_vals[in_bounds]
    y_in = y_vals[in_bounds]
    z_in = z_vals[in_bounds]

    col_idx = np.clip(np.floor((x_in - min_x) / gsd).astype(np.int64), 0, width - 1)
    row_idx = np.clip(np.floor((adjusted_max_y - y_in) / gsd).astype(np.int64), 0, height - 1)

    # 4. Surface aggregation: Maximum elevation per grid cell
    grid = np.full((height, width), -np.inf, dtype=np.float64)
    np.maximum.at(grid, (row_idx, col_idx), z_in)

    # Cells that received no points revert to explicit NoData value
    empty_mask = np.isneginf(grid)
    grid[empty_mask] = nodata_val

    return grid.astype(np.float32), transform, actual_bounds


class DSMExporter:
    """
    Validates and exports Digital Surface Models (DSMs) from reconstructed point clouds.
    """

    @staticmethod
    def pixel_to_coords(row: int, col: int, transform: Affine) -> Tuple[float, float]:
        """
        Converts integer (row, col) pixel indices to center (X, Y) world coordinates in meters.
        Uses direct affine arithmetic, immune to environment-specific tuple wrapper quirks.
        """
        x = transform.c + (col + 0.5) * transform.a
        y = transform.f + (row + 0.5) * transform.e
        return float(x), float(y)

    @staticmethod
    def coords_to_pixel(x: float, y: float, transform: Affine) -> Tuple[int, int]:
        """
        Converts world coordinates (X, Y) in meters to integer (row, col) pixel indices.
        Uses direct affine inversion arithmetic.
        """
        col = int(math.floor((x - transform.c) / transform.a))
        row = int(math.floor((y - transform.f) / transform.e))
        return row, col

    @classmethod
    def export_dsm(
        cls,
        points: np.ndarray,
        geo_meta: CoordinateReferenceMetadata,
        output_path: Union[str, Path],
        config: Optional[DSMConfig] = None,
        allow_unreferenced_local: bool = False,
        allow_exploratory_georeference: bool = False,
        auto_reproject_geographic: bool = False,
    ) -> DSMExportResult:
        """
        Renders a Digital Surface Model and persists it to a valid GeoTIFF raster.

        Args:
            points: (N, 3) point cloud array in target coordinate frame.
            geo_meta: Coordinate reference, datum, and quality metadata.
            output_path: Destination file path (.tif or .geotiff).
            config: Optional DSMConfig rasterization parameters.
            allow_unreferenced_local: If True, permits exporting a local engineering grid
                                      without geospatial claims or fabricated EPSG tags.
            allow_exploratory_georeference: If True, permits exporting projected GeoTIFF
                                            when alignment quality metrics (RMSE, inliers)
                                            are unverified or missing.
            auto_reproject_geographic: If True, automatically reprojects WGS84 geographic
                                       coordinates (degrees) to local UTM (meters).

        Returns:
            DSMExportResult summarizing raster geometry and metadata.
        """
        cfg = config or DSMConfig()
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        active_points = np.asarray(points, dtype=np.float64)

        # 0. Handle Geographic Coordinates Inspection and Reprojection
        if geo_meta.is_georeferenced:
            is_geographic = False
            src_crs_repr = str(geo_meta.crs_epsg or geo_meta.crs_wkt or "UNKNOWN")
            try:
                if geo_meta.crs_epsg is not None:
                    p_crs = CRS.from_epsg(geo_meta.crs_epsg)
                    is_geographic = p_crs.is_geographic
                elif geo_meta.crs_wkt:
                    p_crs = CRS.from_wkt(geo_meta.crs_wkt)
                    is_geographic = p_crs.is_geographic
            except Exception:
                pass

            if is_geographic or geo_meta.horizontal_units.lower() in ["degrees", "degree", "deg"]:
                if auto_reproject_geographic:
                    active_points, target_utm = reproject_geographic_to_utm(
                        active_points,
                        source_crs=geo_meta.crs_epsg or "EPSG:4326"
                    )
                    geo_meta.crs_epsg = target_utm
                    geo_meta.crs_wkt = None
                    geo_meta.horizontal_units = "meters"
                    geo_meta.provenance_note = (
                        f"{geo_meta.provenance_note} [Auto-reprojected from {src_crs_repr} to UTM EPSG:{target_utm}]"
                    ).strip()
                else:
                    raise GeographicCRSError(
                        f"Cannot export DSM with metric GSD in geographic coordinates ({src_crs_repr}, degrees). "
                        "Horizontal coordinates must be meters in a projected CRS. Pass auto_reproject_geographic=True "
                        "or reproject points to UTM before rasterization."
                    )

        # 1. Georeference & CRS Validation
        is_valid_geo, assurance_level, reason = geo_meta.validate_for_georeferenced_export(
            require_accuracy_evidence=not allow_exploratory_georeference
        )

        raster_crs = None
        if not is_valid_geo:
            if assurance_level == "GEOGRAPHIC":
                raise GeographicCRSError(reason)
            elif assurance_level == "INCOMPATIBLE_UNITS":
                raise IncompatibleUnitsError(reason)
            elif assurance_level == "EXPLORATORY":
                raise UnvalidatedAlignmentError(reason)
            elif not allow_unreferenced_local:
                raise UnreferencedGeometryError(
                    f"Refusing to export unverified georeferenced GeoTIFF: {reason}. "
                    "If a local-coordinate model raster is explicitly required without "
                    "geospatial CRS claims, specify allow_unreferenced_local=True."
                )

            # Local coordinate export permitted: strictly purge CRS and geospatial claims
            assurance_tag = "UNREFERENCED_LOCAL_MODEL"
            product_tag = "LOCAL_SURFACE_MODEL_UNREFERENCED"
            crs_tag = "NONE_LOCAL_ARBITRARY"
            provenance_tag = (
                f"UNREFERENCED_LOCAL_COORDINATES: {geo_meta.provenance_note or 'Model-relative scale'}. "
                "DO NOT USE FOR GEOSPATIAL MAPPING OR GIS COMPLIANCE."
            )
            accuracy_tag = "NONE_MODEL_RELATIVE"
            warning_tag = "Model-relative coordinate grid with uncalibrated scale. Do not use for GIS mapping."
            disclaimer_tag = "Model-relative coordinate grid with uncalibrated scale. Not georeferenced or survey validated."
        else:
            product_tag = "DIGITAL_SURFACE_MODEL"
            provenance_tag = geo_meta.provenance_note or "F.O.R.G.E Georeferenced Photogrammetric DSM"

            if geo_meta.crs_epsg is not None:
                raster_crs = CRS.from_epsg(geo_meta.crs_epsg)
                crs_tag = f"EPSG:{geo_meta.crs_epsg}"
            elif geo_meta.crs_wkt:
                raster_crs = CRS.from_wkt(geo_meta.crs_wkt)
                crs_tag = "WKT_CUSTOM"
            else:
                raise UnreferencedGeometryError("Georeferencing marked valid but no CRS definition exists.")

            if assurance_level in ("TELEMETRY_ALIGNED_CONSISTENT", "ACCURACY_VALIDATED"):
                assurance_tag = "TELEMETRY_ALIGNED_CONSISTENT"
                accuracy_tag = "TELEMETRY_INTERNAL_CONSISTENCY_ONLY_NOT_SURVEY_VALIDATED"
                warning_tag = "NONE"
                disclaimer_tag = (
                    "Alignment verified against internal drone telemetry gates (RMSE <= 10m, inliers >= 50%). "
                    "Not an independently surveyed ground-truth accuracy validation; no ground control points (GCPs) applied."
                )
            else:
                assurance_tag = "EXPLORATORY_ESTIMATE_UNVERIFIED_ALIGNMENT"
                accuracy_tag = "NONE_UNVERIFIED_EXPLORATORY_ESTIMATE"
                warning_tag = (
                    "Projected coordinate reference system applied without verified alignment residuals. "
                    "Positional accuracy is uncalibrated and not certified."
                )
                disclaimer_tag = "Uncertified exploratory output; positional accuracy is uncalibrated."

        # 2. Rasterization
        grid, transform, bounds = rasterize_point_cloud_dsm(
            points=active_points,
            gsd=cfg.gsd_m,
            bounds=cfg.bounds,
            nodata_val=cfg.nodata_value
        )

        height, width = grid.shape
        valid_cells = int(np.count_nonzero(grid != cfg.nodata_value))
        nodata_cells = int(width * height - valid_cells)
        coverage_pct = round(100.0 * valid_cells / max(width * height, 1), 2)

        if valid_cells > 0:
            elev_min = float(np.min(grid[grid != cfg.nodata_value]))
            elev_max = float(np.max(grid[grid != cfg.nodata_value]))
        else:
            elev_min = cfg.nodata_value
            elev_max = cfg.nodata_value

        # 3. Write GeoTIFF with explicit metadata
        meta_tags = {
            "PRODUCT_TYPE": product_tag,
            "SURFACE_TYPE": "MAX_ELEVATION_SURFACE",
            "CLASSIFICATION": "UNCLASSIFIED_SURFACE_DSM_NOT_BARE_EARTH_DEM",
            "UNITS": geo_meta.vertical_units,
            "HORIZONTAL_UNITS": geo_meta.horizontal_units,
            "GSD_METERS": f"{cfg.gsd_m:.4f}",
            "NO_DATA_VALUE": str(cfg.nodata_value),
            "GEOREFERENCED": "TRUE" if is_valid_geo else "FALSE",
            "GEOSPATIAL_ASSURANCE": assurance_tag,
            "ACCURACY_CLAIM": accuracy_tag,
            "ACCURACY_DISCLAIMER": disclaimer_tag,
            "COORDINATE_SYSTEM": crs_tag,
            "VERTICAL_DATUM": geo_meta.vertical_datum,
            "VERTICAL_DATUM_NOTE": geo_meta.vertical_datum_note,
            "PROVENANCE": provenance_tag,
            "SOFTWARE": "F.O.R.G.E. OpenCV AI Competition 2026",
        }

        if warning_tag != "NONE":
            meta_tags["GEOSPATIAL_WARNING"] = warning_tag
        if geo_meta.sim3_scale is not None:
            meta_tags["SIM3_SCALE"] = f"{geo_meta.sim3_scale:.6f}"
        if geo_meta.alignment_rmse_m is not None:
            meta_tags["ALIGNMENT_RMSE_M"] = f"{geo_meta.alignment_rmse_m:.4f}"
        if geo_meta.alignment_inlier_ratio is not None:
            meta_tags["ALIGNMENT_INLIER_RATIO"] = f"{geo_meta.alignment_inlier_ratio:.4f}"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": cfg.dtype,
            "crs": raster_crs,
            "transform": transform,
            "nodata": cfg.nodata_value,
            "compress": "deflate",
        }

        # Enable tiling only for large datasets (>= 256x256) where multiple-of-16 block requirements are met
        if width >= 256 and height >= 256:
            profile["tiled"] = True
            profile["blockxsize"] = 256
            profile["blockysize"] = 256

        with rasterio.open(str(out_file), "w", **profile) as dst:
            dst.write(grid, 1)
            dst.update_tags(**meta_tags)

        status_msg = (
            f"Successfully exported {product_tag} ({width}x{height}, GSD={cfg.gsd_m}m, Assurance={assurance_tag}) "
            f"to {out_file.name}. Coverage: {coverage_pct}% ({valid_cells} valid cells)."
        )

        return DSMExportResult(
            success=True,
            output_path=str(out_file),
            is_georeferenced=is_valid_geo,
            assurance_level=assurance_tag,
            crs_string=crs_tag,
            raster_width=width,
            raster_height=height,
            gsd=cfg.gsd_m,
            bounds=bounds,
            elevation_min=elev_min,
            elevation_max=elev_max,
            valid_cell_count=valid_cells,
            nodata_cell_count=nodata_cells,
            valid_coverage_pct=coverage_pct,
            metadata=meta_tags,
            status_message=status_msg
        )
