import type { SceneEntry, QualitySummary } from '../types';

export const SAMPLE_QUALITY_SUMMARY: QualitySummary = {
  overall_status: 'PARTIAL',
  vertex_count: 184,
  triangle_count: 244,
  bounding_box_meters: [4.5, 2.7, 4.0],
  degenerate_face_ratio: 0.0,
  viewer_load_verified: true,
  visual_inspection_passed: true,
  known_limitations:
    'Synthetic indoor test fixture for visual development only. Real reconstruction accuracy requires calibrated camera intrinsics and continuous temporal corridor coverage.',
  coordinate_system: 'Metric coordinate frame (meters). Origin at room center floor [0, 0, 0].',
  gates: [
    {
      gate: 'Gate A',
      title: 'Process Execution & Output Manifest',
      status: 'PASS',
      summary: 'Pipeline exited with code 0. Required output files (OBJ, MTL, PLY) present and non-empty.',
      details: {
        exit_code: 0,
        manifest_files: ['indoor_sample_room.obj', 'indoor_sample_room.mtl', 'indoor_sample_room.ply'],
        total_output_bytes: 24836,
      },
    },
    {
      gate: 'Gate B',
      title: 'File Integrity & Format Parsing',
      status: 'PASS',
      summary: 'Parsed without errors. Valid OBJ face indices, defined normals, and correct material references.',
      details: {
        obj_readable: true,
        mtl_referenced: 'indoor_sample_room.mtl',
        ply_ascii_valid: true,
      },
    },
    {
      gate: 'Gate C',
      title: 'Geometry Diagnostics & Bounding Extents',
      status: 'PASS',
      summary: 'All vertex coordinates finite. Extents [4.5m x 2.7m x 4.0m]. Degenerate face count: 0 (0.0%).',
      details: {
        vertices: 184,
        faces: 244,
        extents_m: [4.5, 2.7, 4.0],
        degenerate_faces: 0,
        non_finite_coords: 0,
      },
    },
    {
      gate: 'Gate D',
      title: 'Viewer Loading & Camera Framing',
      status: 'PASS',
      summary: 'Rendered in WebGL viewport. Camera framed to center [0, 1.35, 0]. No shader or texture errors.',
      details: {
        threejs_webgl_ok: true,
        framing_distance_m: 5.2,
      },
    },
    {
      gate: 'Gate E',
      title: 'Visual Review & Environmental Context',
      status: 'WARN',
      summary: 'SAMPLE FIXTURE — Visually coherent office room geometry. Doorway and furniture distinct. Not from video.',
      details: {
        is_sample_asset: true,
        unseen_facades_synthesized: false,
      },
    },
  ],
};

export const SAMPLE_INDOOR_SCENE: SceneEntry = {
  id: 'sample-indoor-office-01',
  display_name: 'Sample Indoor Office Space',
  subtitle: '4.5m x 4.0m Study with Desk & Bookshelf',
  is_sample_asset: true,
  sample_watermark_text: 'SAMPLE TEST FIXTURE — VISUAL VALIDATION ONLY (NOT A REAL VIDEO RECONSTRUCTION)',
  acceptance_status: 'PARTIAL',
  known_limitations:
    'Seeded fixture for UI development and viewer verification. Do not represent as a real video reconstruction.',
  coordinate_status: 'Metric scale (meters). Floor centered at [0, 0, 0].',
  vertex_count: 184,
  triangle_count: 244,
  default_representation: 'textured_obj',
  representations: {
    textured_obj: {
      type: 'obj',
      label: 'Material Colored OBJ',
      obj_url: '/sample/indoor_sample_room.obj',
      mtl_url: '/sample/indoor_sample_room.mtl',
      file_size_bytes: 14387,
      available: true,
    },
    photographic_ply: {
      type: 'ply',
      label: 'Photographic Vertex PLY',
      ply_url: '/sample/indoor_sample_room.ply',
      file_size_bytes: 9786,
      available: true,
    },
    untextured_ply: {
      type: 'ply',
      label: 'Raw Geometry PLY',
      ply_url: '/sample/indoor_sample_room.ply',
      file_size_bytes: 9786,
      available: true,
    },
  },
  initial_camera: {
    position: [0.0, 2.5, 4.5],
    target: [0.0, 1.2, 0.0],
  },
  quality_summary: SAMPLE_QUALITY_SUMMARY,
};

