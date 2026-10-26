"""
Unit Tests for F.O.R.G.E. Digital Surface Model (DSM) GeoTIFF Exporter.

Comprehensive Correctness Audit & Regression Suite:
1. Derivation of WGS84 UTM zones across hemispheres.
2. Rejection of unreferenced/model-relative geometry by default.
3. Rejection of geographic CRS (EPSG:4326) with metric GSD (no silent degree-as-meter treating).
4. Automatic reprojection from WGS84 geographic (lon, lat) to UTM metric grid.
5. Incompatible horizontal/vertical units rejection (raising IncompatibleUnitsError).
6. Rejection of unvalidated georeferencing claims when alignment quality evidence (RMSE, inliers) is missing.
7. Explicit exploratory georeferenced export when requested, verifying unverified accuracy tags.
8. Rigorous accuracy-validated georeferenced export with full telemetry residuals and vertical datum notes.
9. Safe local-coordinate raster export without fabricated EPSG or geospatial claims.
10. Exact elevation reproduction on synthetic planar ramps and stepped structures (Z_max).
11. Strict NoData handling for unobserved/empty cells.
12. Resilience against non-finite (NaN, Inf) inputs and degenerate bounds/GSD.
13. Standardized GIS tags explicitly distinguishing DSM from bare-earth DEM.
14. Non-4326 geographic CRS rejection (e.g. EPSG:4269 NAD83).
15. Exact affine-transform roundtrip, pixel-to-coords, and raster bounds accuracy.
16. Degenerate bounds and single-point cloud handling.
"""

import os
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

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
    rasterize_point_cloud_dsm,
)


class TestDSMGeoTIFFExporter(unittest.TestCase):
    """Test suite for DSM GeoTIFF generation, CRS verification, and geospatial validation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_derive_utm_crs_from_latlon(self):
        """Test standard WGS84 UTM zone EPSG calculation across hemispheres."""
        # Delhi, India (Northern hemisphere, ~77.2 E -> Zone 43N)
        epsg_delhi = derive_utm_crs_from_latlon(28.6139, 77.2090)
        self.assertEqual(epsg_delhi, 32643)

        # Mumbai, India (Northern hemisphere, ~72.8 E -> Zone 43N)
        epsg_mumbai = derive_utm_crs_from_latlon(19.0760, 72.8777)
        self.assertEqual(epsg_mumbai, 32643)

        # Sydney, Australia (Southern hemisphere, ~151.2 E -> Zone 56S)
        epsg_sydney = derive_utm_crs_from_latlon(-33.8688, 151.2093)
        self.assertEqual(epsg_sydney, 32756)

        # Invalid bounds
        with self.assertRaises(ValueError):
            derive_utm_crs_from_latlon(95.0, 77.0)
        with self.assertRaises(ValueError):
            derive_utm_crs_from_latlon(28.0, 200.0)

    def test_02_rejection_of_unreferenced_geometry(self):
        """Unreferenced models (e.g. Campus_0049) must be rejected by default."""
        pts = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.2],
            [0.0, 1.0, 1.1],
            [1.0, 1.0, 1.3],
        ], dtype=np.float64)

        unref_meta = CoordinateReferenceMetadata(
            is_georeferenced=False,
            horizontal_units="model_relative_arbitrary",
            vertical_units="model_relative_arbitrary",
            provenance_note="Campus 0049 uncalibrated model-relative scale"
        )

        out_path = self.out_dir / "unreferenced_test.tif"

        # Default export MUST reject without allow_unreferenced_local=True
        with self.assertRaises(UnreferencedGeometryError) as ctx:
            DSMExporter.export_dsm(pts, unref_meta, out_path, allow_unreferenced_local=False)
        self.assertIn("non-georeferenced", str(ctx.exception))
        self.assertFalse(out_path.exists())

    def test_03_rejection_of_geographic_crs_without_reprojection(self):
        """Geographic coordinates (EPSG:4326, degrees) must be rejected with metric GSD."""
        # Delhi coordinates in degrees: lon=77.209, lat=28.613
        pts_geo = np.array([
            [77.2090, 28.6139, 215.0],
            [77.2095, 28.6142, 216.0],
        ], dtype=np.float64)

        geo_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=4326,  # WGS84 Geographic 2D
            horizontal_units="degrees",
            vertical_units="meters",
            sim3_scale=1.0,
            alignment_rmse_m=0.5,
            alignment_inlier_ratio=0.9
        )

        out_path = self.out_dir / "geographic_fail.tif"
        # Must reject by default: degrees cannot be silently treated as meters
        with self.assertRaises(GeographicCRSError) as ctx:
            DSMExporter.export_dsm(pts_geo, geo_meta, out_path, config=DSMConfig(gsd_m=0.25))
        self.assertIn("geographic coordinates", str(ctx.exception))
        self.assertFalse(out_path.exists())

    def test_04_automatic_reprojection_from_geographic_to_utm(self):
        """When auto_reproject_geographic=True, coordinates must convert to local UTM."""
        pts_geo = np.array([
            [77.2090, 28.6139, 215.0],
            [77.2095, 28.6142, 216.0],
        ], dtype=np.float64)

        geo_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=4326,
            horizontal_units="degrees",
            vertical_units="meters",
            sim3_scale=1.0,
            alignment_rmse_m=0.5,
            alignment_inlier_ratio=0.9,
            provenance_note="GPS raw points"
        )

        out_path = self.out_dir / "reprojected_utm_dsm.tif"
        res = DSMExporter.export_dsm(
            pts_geo, geo_meta, out_path,
            config=DSMConfig(gsd_m=1.0),
            auto_reproject_geographic=True
        )

        self.assertTrue(res.success)
        self.assertEqual(res.crs_string, "EPSG:32643")  # Delhi lands in UTM 43N
        self.assertTrue(out_path.is_file())

        with rasterio.open(out_path) as src:
            self.assertEqual(src.crs.to_epsg(), 32643)
            # Verify coordinates are in UTM meter range (>500,000 Easting, >2,000,000 Northing)
            self.assertGreater(res.bounds[0], 500000.0)
            self.assertGreater(src.transform.c, 500000.0)
            self.assertGreater(src.transform.f, 2000000.0)

    def test_05_rejection_of_incompatible_units(self):
        """Rejects non-meter horizontal or vertical units with IncompatibleUnitsError."""
        pts = np.array([[100.0, 200.0, 10.0], [105.0, 205.0, 12.0]], dtype=np.float64)

        # Incompatible horizontal units
        bad_h_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="feet",  # Incompatible with metric GSD
            vertical_units="meters",
            alignment_rmse_m=0.5,
            alignment_inlier_ratio=0.9
        )
        with self.assertRaises(IncompatibleUnitsError) as ctx:
            DSMExporter.export_dsm(pts, bad_h_meta, self.out_dir / "bad_h.tif")
        self.assertIn("horizontal units", str(ctx.exception))
        # Also confirm it is a subclass of UnreferencedGeometryError
        self.assertTrue(issubclass(IncompatibleUnitsError, UnreferencedGeometryError))

        # Incompatible vertical units
        bad_v_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="feet",  # Incompatible with metric elevations
            alignment_rmse_m=0.5,
            alignment_inlier_ratio=0.9
        )
        with self.assertRaises(IncompatibleUnitsError) as ctx:
            DSMExporter.export_dsm(pts, bad_v_meta, self.out_dir / "bad_v.tif")
        self.assertIn("vertical units", str(ctx.exception))

    def test_06_rejection_of_missing_alignment_evidence(self):
        """Standard georeferenced export must reject unverified claims when RMSE/inliers are missing."""
        pts = np.array([[100.0, 200.0, 10.0], [105.0, 205.0, 12.0]], dtype=np.float64)

        # Missing RMSE and inliers
        unverified_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            sim3_scale=1.0,
            alignment_rmse_m=None,      # Missing evidence
            alignment_inlier_ratio=None  # Missing evidence
        )

        out_path = self.out_dir / "unverified_fail.tif"
        # Must reject by default because positional accuracy cannot be certified
        with self.assertRaises(UnvalidatedAlignmentError) as ctx:
            DSMExporter.export_dsm(pts, unverified_meta, out_path, allow_exploratory_georeference=False)
        self.assertIn("Missing alignment quality evidence", str(ctx.exception))
        self.assertFalse(out_path.exists())

    def test_07_exploratory_georeferenced_export_with_explicit_flag(self):
        """When allow_exploratory_georeference=True, unverified models export with exploratory tags."""
        pts = np.array([[100.0, 200.0, 10.0], [105.0, 205.0, 12.0]], dtype=np.float64)

        unverified_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            sim3_scale=1.0,
            alignment_rmse_m=None,
            alignment_inlier_ratio=None
        )

        out_path = self.out_dir / "exploratory_success.tif"
        res = DSMExporter.export_dsm(
            pts, unverified_meta, out_path,
            allow_exploratory_georeference=True
        )

        self.assertTrue(res.success)
        self.assertEqual(res.assurance_level, "EXPLORATORY_ESTIMATE_UNVERIFIED_ALIGNMENT")

        with rasterio.open(out_path) as src:
            tags = src.tags()
            self.assertEqual(tags.get("GEOSPATIAL_ASSURANCE"), "EXPLORATORY_ESTIMATE_UNVERIFIED_ALIGNMENT")
            self.assertEqual(tags.get("ACCURACY_CLAIM"), "NONE_UNVERIFIED_EXPLORATORY_ESTIMATE")
            self.assertIn("without verified alignment residuals", tags.get("GEOSPATIAL_WARNING", ""))

    def test_08_rejection_of_unusable_alignment_quality(self):
        """Georeferenced claims with excessive RMSE or poor inliers must be rejected."""
        pts = np.array([[100.0, 200.0, 10.0], [105.0, 205.0, 12.0]], dtype=np.float64)

        # Metadata with high RMSE uncertainty (25m error)
        bad_rmse_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            alignment_rmse_m=25.0,  # Exceeds 10m threshold
            alignment_inlier_ratio=0.85,
            sim3_scale=1.0
        )
        with self.assertRaises(UnvalidatedAlignmentError) as ctx:
            DSMExporter.export_dsm(pts, bad_rmse_meta, self.out_dir / "bad_rmse.tif")
        self.assertIn("residual RMSE", str(ctx.exception))

        # Metadata with poor inlier ratio (15%)
        bad_inliers_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            alignment_rmse_m=1.2,
            alignment_inlier_ratio=0.15,  # Below 50% floor
            sim3_scale=1.0
        )
        with self.assertRaises(UnvalidatedAlignmentError) as ctx:
            DSMExporter.export_dsm(pts, bad_inliers_meta, self.out_dir / "bad_inliers.tif")
        self.assertIn("inlier ratio", str(ctx.exception))

        # Degenerate scale factor
        bad_scale_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            sim3_scale=-0.5
        )
        with self.assertRaises(UnreferencedGeometryError) as ctx:
            DSMExporter.export_dsm(pts, bad_scale_meta, self.out_dir / "bad_scale.tif")
        self.assertIn("scale factor", str(ctx.exception))

    def test_09_allow_unreferenced_local_export(self):
        """When explicitly permitted, local rasters must not contain fabricated EPSG tags."""
        pts = np.array([
            [0.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [0.0, 4.0, 2.0],
            [4.0, 4.0, 2.0],
        ], dtype=np.float64)

        local_meta = CoordinateReferenceMetadata(
            is_georeferenced=False,
            horizontal_units="model_relative",
            vertical_units="model_relative",
            provenance_note="Visual inspection proxy only"
        )

        out_path = self.out_dir / "local_grid.tif"
        res = DSMExporter.export_dsm(
            pts, local_meta, out_path,
            config=DSMConfig(gsd_m=1.0),
            allow_unreferenced_local=True
        )

        self.assertTrue(res.success)
        self.assertFalse(res.is_georeferenced)
        self.assertEqual(res.crs_string, "NONE_LOCAL_ARBITRARY")
        self.assertEqual(res.assurance_level, "UNREFERENCED_LOCAL_MODEL")
        self.assertTrue(out_path.is_file())

        with rasterio.open(out_path) as src:
            self.assertIsNone(src.crs)
            tags = src.tags()
            self.assertEqual(tags.get("GEOREFERENCED"), "FALSE")
            self.assertEqual(tags.get("COORDINATE_SYSTEM"), "NONE_LOCAL_ARBITRARY")
            self.assertEqual(tags.get("PRODUCT_TYPE"), "LOCAL_SURFACE_MODEL_UNREFERENCED")
            self.assertEqual(tags.get("GEOSPATIAL_ASSURANCE"), "UNREFERENCED_LOCAL_MODEL")
            self.assertIn("UNREFERENCED_LOCAL_COORDINATES", tags.get("PROVENANCE", ""))

    def test_10_synthetic_planar_ramp_surface(self):
        """Test exact elevation reproduction on a metric tilted planar surface."""
        # Spatial extent: [100.0, 110.0] x [200.0, 210.0] -> 10m x 10m
        # With GSD = 0.5m, raster size should be exactly 20 x 20 pixels
        grid_x, grid_y = np.meshgrid(
            np.linspace(100.0, 110.0, 101),
            np.linspace(200.0, 210.0, 101)
        )
        x_flat = grid_x.flatten()
        y_flat = grid_y.flatten()
        z_flat = 0.5 * x_flat + 0.2 * y_flat + 10.0
        pts = np.column_stack([x_flat, y_flat, z_flat])

        meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            sim3_scale=1.0,
            alignment_rmse_m=0.05,
            alignment_inlier_ratio=1.0,
            provenance_note="Synthetic planarity calibration"
        )

        out_path = self.out_dir / "planar_ramp_dsm.tif"
        cfg = DSMConfig(gsd_m=0.5, nodata_value=-9999.0)

        result = DSMExporter.export_dsm(pts, meta, out_path, config=cfg)

        self.assertTrue(result.success)
        self.assertEqual(result.raster_width, 20)
        self.assertEqual(result.raster_height, 20)
        self.assertEqual(result.nodata_cell_count, 0)
        self.assertEqual(result.valid_coverage_pct, 100.0)

        with rasterio.open(out_path) as src:
            self.assertEqual(src.crs.to_epsg(), 32643)
            self.assertEqual(src.nodata, -9999.0)
            self.assertEqual(src.width, 20)
            self.assertEqual(src.height, 20)

            # Check affine transform resolution
            self.assertAlmostEqual(src.transform.a, 0.5, places=5)
            self.assertAlmostEqual(src.transform.e, -0.5, places=5)
            self.assertAlmostEqual(src.transform.c, 100.0, places=4)
            self.assertAlmostEqual(src.transform.f, 210.0, places=4)

            band = src.read(1)
            for r in [0, 5, 10, 19]:
                for c in [0, 5, 10, 19]:
                    cell_x, cell_y = DSMExporter.pixel_to_coords(r, c, src.transform)
                    val = band[r, c]
                    expected_plane_z = 0.5 * cell_x + 0.2 * cell_y + 10.0
                    self.assertAlmostEqual(val, expected_plane_z, delta=0.5)

    def test_11_synthetic_stepped_surface_max_elevation(self):
        """A DSM must record the top-most reflective surface elevation, never the lower ground."""
        pts = np.array([
            [2.2, 2.2, 5.0],   # Ground point
            [2.3, 2.3, 5.1],   # Ground point
            [2.2, 2.2, 14.5],  # Rooftop fixture
            [2.3, 2.3, 14.2],  # Rooftop fixture
        ], dtype=np.float64)

        meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            sim3_scale=1.0,
            alignment_rmse_m=0.1,
            alignment_inlier_ratio=0.9
        )

        out_path = self.out_dir / "stepped_dsm.tif"
        cfg = DSMConfig(gsd_m=1.0, bounds=(0.0, 0.0, 5.0, 5.0))
        result = DSMExporter.export_dsm(pts, meta, out_path, config=cfg)

        with rasterio.open(out_path) as src:
            band = src.read(1)
            row, col = DSMExporter.coords_to_pixel(2.2, 2.2, src.transform)
            binned_val = band[row, col]
            # Must equal the maximum surface elevation 14.5m, not ground 5.0m
            self.assertEqual(binned_val, 14.5)

    def test_12_empty_cells_and_nodata_handling(self):
        """Unobserved cells must strictly receive the sentinel NoData value."""
        pts = np.array([
            [0.1, 0.1, 10.0],
            [9.9, 0.1, 10.0],
            [0.1, 9.9, 10.0],
            [9.9, 9.9, 10.0],
        ], dtype=np.float64)

        meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            alignment_rmse_m=0.2,
            alignment_inlier_ratio=0.95
        )

        out_path = self.out_dir / "nodata_test.tif"
        cfg = DSMConfig(gsd_m=1.0, bounds=(0.0, 0.0, 10.0, 10.0), nodata_value=-9999.0)
        result = DSMExporter.export_dsm(pts, meta, out_path, config=cfg)

        self.assertEqual(result.raster_width, 10)
        self.assertEqual(result.raster_height, 10)
        self.assertEqual(result.valid_cell_count, 4)
        self.assertEqual(result.nodata_cell_count, 96)
        self.assertEqual(result.valid_coverage_pct, 4.0)

        with rasterio.open(out_path) as src:
            band = src.read(1)
            self.assertEqual(band[5, 5], -9999.0)
            r_top_left, c_top_left = DSMExporter.coords_to_pixel(0.1, 9.9, src.transform)
            self.assertEqual(band[r_top_left, c_top_left], 10.0)

    def test_13_non_finite_and_invalid_inputs(self):
        """Non-finite points must be pruned, and completely empty inputs cleanly rejected."""
        pts_with_nans = np.array([
            [1.0, 1.0, 10.0],
            [np.nan, 2.0, 10.0],
            [2.0, np.inf, 10.0],
            [2.0, 2.0, -np.inf],
            [3.0, 3.0, 12.0],
        ], dtype=np.float64)

        grid, transform, bounds = rasterize_point_cloud_dsm(pts_with_nans, gsd=1.0)
        valid_vals = grid[grid != -9999.0]
        self.assertEqual(len(valid_vals), 2)
        self.assertIn(10.0, valid_vals)
        self.assertIn(12.0, valid_vals)

        # All NaNs
        with self.assertRaises(ValueError):
            rasterize_point_cloud_dsm(np.full((5, 3), np.nan), gsd=1.0)

        # Invalid GSD
        with self.assertRaises(InvalidRasterConfigurationError):
            rasterize_point_cloud_dsm(np.array([[1.0, 2.0, 3.0]]), gsd=-0.5)

        # Reversed bounds
        with self.assertRaises(InvalidRasterConfigurationError):
            rasterize_point_cloud_dsm(
                np.array([[1.0, 2.0, 3.0]]), gsd=1.0, bounds=(10.0, 10.0, 5.0, 5.0)
            )

    def test_14_metadata_and_vertical_datum_tags(self):
        """Standardized GIS tags must explicitly classify the file as DSM and document vertical datum."""
        pts = np.array([[10.0, 10.0, 100.0], [20.0, 20.0, 105.0]], dtype=np.float64)
        meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            sim3_scale=1.045,
            alignment_rmse_m=0.34,
            alignment_inlier_ratio=0.92,
            provenance_note="River Bridge georeferenced survey flight",
            vertical_datum="WGS84_ELLIPSOIDAL",
            vertical_datum_note="Ellipsoidal height; no geoid undulation applied"
        )

        out_path = self.out_dir / "tagged_dsm.tif"
        res = DSMExporter.export_dsm(pts, meta, out_path, config=DSMConfig(gsd_m=1.0))

        with rasterio.open(out_path) as src:
            tags = src.tags()
            self.assertEqual(tags.get("PRODUCT_TYPE"), "DIGITAL_SURFACE_MODEL")
            self.assertEqual(tags.get("SURFACE_TYPE"), "MAX_ELEVATION_SURFACE")
            self.assertEqual(tags.get("CLASSIFICATION"), "UNCLASSIFIED_SURFACE_DSM_NOT_BARE_EARTH_DEM")
            self.assertEqual(tags.get("UNITS"), "meters")
            self.assertEqual(tags.get("HORIZONTAL_UNITS"), "meters")
            self.assertEqual(tags.get("GEOREFERENCED"), "TRUE")
            self.assertEqual(tags.get("GEOSPATIAL_ASSURANCE"), "TELEMETRY_ALIGNED_CONSISTENT")
            self.assertEqual(tags.get("ACCURACY_CLAIM"), "TELEMETRY_INTERNAL_CONSISTENCY_ONLY_NOT_SURVEY_VALIDATED")
            self.assertIn("Not an independently surveyed ground-truth", tags.get("ACCURACY_DISCLAIMER", ""))
            self.assertEqual(tags.get("COORDINATE_SYSTEM"), "EPSG:32643")
            self.assertEqual(tags.get("VERTICAL_DATUM"), "WGS84_ELLIPSOIDAL")
            self.assertEqual(tags.get("VERTICAL_DATUM_NOTE"), "Ellipsoidal height; no geoid undulation applied")
            self.assertEqual(tags.get("SOFTWARE"), "F.O.R.G.E. OpenCV AI Competition 2026")
            self.assertEqual(tags.get("SIM3_SCALE"), "1.045000")
            self.assertEqual(tags.get("ALIGNMENT_RMSE_M"), "0.3400")
            self.assertEqual(tags.get("ALIGNMENT_INLIER_RATIO"), "0.9200")

    def test_15_non_4326_geographic_crs_rejection(self):
        """Any geographic CRS (e.g. EPSG:4269 NAD83) must raise GeographicCRSError with metric GSD."""
        pts_geo = np.array([
            [-105.2705, 40.0150, 1624.0],  # Boulder, Colorado in degrees
            [-105.2700, 40.0155, 1625.0],
        ], dtype=np.float64)

        nad83_meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=4269,  # NAD83 Geographic 2D
            horizontal_units="degrees",
            vertical_units="meters",
            alignment_rmse_m=0.3,
            alignment_inlier_ratio=0.95
        )

        out_path = self.out_dir / "nad83_fail.tif"
        with self.assertRaises(GeographicCRSError) as ctx:
            DSMExporter.export_dsm(pts_geo, nad83_meta, out_path, config=DSMConfig(gsd_m=0.5))
        self.assertIn("geographic coordinates", str(ctx.exception))
        self.assertFalse(out_path.exists())

    def test_16_affine_transform_roundtrip_and_bounds_accuracy(self):
        """Verifies exact affine transform resolution, pixel-to-coords roundtrip, and extent alignment."""
        pts = np.array([
            [500.0, 1000.0, 50.0],
            [550.0, 1050.0, 55.0],
        ], dtype=np.float64)

        meta = CoordinateReferenceMetadata(
            is_georeferenced=True,
            crs_epsg=32643,
            horizontal_units="meters",
            vertical_units="meters",
            alignment_rmse_m=0.2,
            alignment_inlier_ratio=0.95
        )

        gsd = 0.5
        out_path = self.out_dir / "transform_bounds_test.tif"
        res = DSMExporter.export_dsm(pts, meta, out_path, config=DSMConfig(gsd_m=gsd))

        self.assertTrue(res.success)
        # Bounding box should span [500.0, 1000.0] to [550.0, 1050.0]
        # Dimensions: 50.0 / 0.5 = 100 pixels wide and high
        self.assertEqual(res.raster_width, 100)
        self.assertEqual(res.raster_height, 100)
        self.assertEqual(res.bounds, (500.0, 1000.0, 550.0, 1050.0))

        with rasterio.open(out_path) as src:
            tf = src.transform
            # Pixel size
            self.assertAlmostEqual(tf.a, gsd)
            self.assertAlmostEqual(tf.e, -gsd)
            # Origin (top-left)
            self.assertAlmostEqual(tf.c, 500.0)
            self.assertAlmostEqual(tf.f, 1050.0)

            # Roundtrip test across all corners and center
            test_indices = [(0, 0), (0, 99), (99, 0), (99, 99), (50, 50), (25, 75)]
            for r, c in test_indices:
                x, y = DSMExporter.pixel_to_coords(r, c, tf)
                r_back, c_back = DSMExporter.coords_to_pixel(x, y, tf)
                self.assertEqual((r_back, c_back), (r, c), f"Roundtrip mismatch at pixel ({r}, {c})")

            # Check corner cell centers
            top_left_x, top_left_y = DSMExporter.pixel_to_coords(0, 0, tf)
            self.assertAlmostEqual(top_left_x, 500.0 + 0.5 * gsd)
            self.assertAlmostEqual(top_left_y, 1050.0 - 0.5 * gsd)

            bottom_right_x, bottom_right_y = DSMExporter.pixel_to_coords(99, 99, tf)
            self.assertAlmostEqual(bottom_right_x, 550.0 - 0.5 * gsd)
            self.assertAlmostEqual(bottom_right_y, 1000.0 + 0.5 * gsd)

    def test_17_single_point_cloud_and_degenerate_bounds(self):
        """Single point clouds should produce a valid 1x1 cell raster, while inverted bounds fail cleanly."""
        single_pt = np.array([[250.0, 750.0, 42.0]], dtype=np.float64)

        grid, transform, bounds = rasterize_point_cloud_dsm(single_pt, gsd=1.0)
        self.assertEqual(grid.shape, (1, 1))
        self.assertEqual(grid[0, 0], 42.0)
        self.assertEqual(bounds, (250.0, 750.0, 251.0, 751.0))

        # Inverted X bounds
        with self.assertRaises(InvalidRasterConfigurationError):
            rasterize_point_cloud_dsm(single_pt, gsd=1.0, bounds=(300.0, 700.0, 200.0, 800.0))

        # Inverted Y bounds
        with self.assertRaises(InvalidRasterConfigurationError):
            rasterize_point_cloud_dsm(single_pt, gsd=1.0, bounds=(200.0, 800.0, 300.0, 700.0))


if __name__ == "__main__":
    unittest.main()
