import cv2, numpy as np
from pathlib import Path

v12_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12')
v11_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v11')

W, H = 2000, 1350
canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

# Header
cv2.putText(canvas, 'F.O.R.G.E V12 - Neutral Geometry & Surface Quality Comparative Audit', (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.05, (15, 15, 15), 2, cv2.LINE_AA)
cv2.putText(canvas, 'Evaluating Surface Continuity, Planarity, and Topological Integrity across Candidates C (V9), A (Poisson), and B (Delaunay)', (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 90, 90), 1, cv2.LINE_AA)
cv2.line(canvas, (40, 130), (1960, 130), (220, 220, 220), 1)

# Load renders
im_v9 = cv2.imread(str(v11_dir / 'v9_render_oblique.png'))
im_poisson = cv2.imread(str(v12_dir / 'v12_poisson_oblique.png'))
im_delaunay = cv2.imread(str(v12_dir / 'v12_delaunay_oblique.png'))

panels = [
    {
        'title': 'Candidate C - V9 Reference Mesh',
        'sub': 'Monocular Relative Depth Fusion Baseline',
        'img': im_v9,
        'status': 'RETAINED VISUAL BASELINE',
        'status_color': (0, 140, 0),
        'metrics': [
            'Triangle Count: 1,386,245 faces (695k vertices)',
            'Planarity: Smooth, coherent planar roof & road surfaces',
            'Continuity: Recognizable architectural & campus structure',
            'Defects: Squashed elevation (0.58m), non-manifold edges',
            'Verdict: Preserved as existing visual baseline'
        ],
        'x': 50
    },
    {
        'title': 'Candidate A - Native COLMAP Poisson',
        'sub': 'Screened Poisson Surface Reconstruction (Depth 10)',
        'img': im_poisson,
        'status': 'REJECTED: NORMAL POLARITY TEARING',
        'status_color': (0, 0, 180),
        'metrics': [
            'Triangle Count: 188,774 faces (Trim 10) / 412k (Trim 0)',
            'Defect Mechanism: Opposing normal orientations in cloud',
            'Polarity Inversion: 26.9% upward normals vs 73.1% downward',
            'Topological Failure: Indicator field cancels, hollowing roof',
            'Verdict: Rejected; surface torn into ribbons & hollow loops'
        ],
        'x': 690
    },
    {
        'title': 'Candidate B - Native COLMAP Delaunay',
        'sub': 'Graph-Cut Delaunay Visibility Meshing',
        'img': im_delaunay,
        'status': 'REJECTED: UNBOUNDED HORIZON SLIVERS',
        'status_color': (0, 100, 200),
        'metrics': [
            'Triangle Count: 204,875 faces (101,739 vertices)',
            'Defect Mechanism: Unbounded horizon tetrahedralization',
            'Boundary Leaks: Distant sky rays bridge to foreground',
            'Topological Failure: Non-manifold geometry & giant slivers',
            'Verdict: Rejected; non-planar roof & irregular edge spikes'
        ],
        'x': 1330
    }
]

card_w = 620
card_h = 1000
top_y = 160

for p in panels:
    ox = p['x']
    cv2.rectangle(canvas, (ox, top_y), (ox + card_w, top_y + card_h), (225, 225, 225), 1)
    cv2.rectangle(canvas, (ox, top_y), (ox + card_w, top_y + 75), (248, 248, 248), -1)
    cv2.line(canvas, (ox, top_y + 75), (ox + card_w, top_y + 75), (220, 220, 220), 1)
    cv2.putText(canvas, p['title'], (ox + 15, top_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, p['sub'], (ox + 15, top_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 100), 1, cv2.LINE_AA)

    if p['img'] is not None:
        tgt_w = card_w - 20
        tgt_h = int(tgt_w * 450 / 600)
        im_res = cv2.resize(p['img'], (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        canvas[top_y + 85 : top_y + 85 + tgt_h, ox + 10 : ox + 10 + tgt_w] = im_res
        cv2.rectangle(canvas, (ox + 10, top_y + 85), (ox + 10 + tgt_w, top_y + 85 + tgt_h), (235, 235, 235), 1)

    sy = top_y + 85 + tgt_h + 30
    cv2.rectangle(canvas, (ox + 15, sy - 20), (ox + card_w - 15, sy + 15), (245, 245, 245), -1)
    cv2.rectangle(canvas, (ox + 15, sy - 20), (ox + card_w - 15, sy + 15), p['status_color'], 1)
    cv2.putText(canvas, p['status'], (ox + 25, sy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, p['status_color'], 2, cv2.LINE_AA)

    my = sy + 45
    for line in p['metrics']:
        bullet = '  * '
        cv2.putText(canvas, bullet + line, (ox + 15, my), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
        my += 34

by = top_y + card_h + 25
cv2.rectangle(canvas, (40, by), (1960, by + 120), (246, 248, 252), -1)
cv2.rectangle(canvas, (40, by), (1960, by + 120), (200, 215, 235), 1)
cv2.putText(canvas, 'MANDATORY SECTION 5 & 9 QUALITY GATE ENFORCEMENT', (60, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 50, 130), 2, cv2.LINE_AA)
cv2.putText(canvas, '1. Quality Gate Mandate: Do not texture or export a final GLB until the neutral mesh passes geometry inspection.', (60, by + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
cv2.putText(canvas, '2. Native Meshing Assessment: Poisson meshing tore due to un-oriented point normals; Delaunay produced non-manifold horizon slivers.', (60, by + 87), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
cv2.putText(canvas, '3. Decision: Texture baking and GLB export are WITHHELD. V9 remains the active visual baseline. Fused cloud (374k pts) preserved for mesh refinement.', (60, by + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)

out_path = v12_dir / 'neutral_geometry_preview.png'
cv2.imwrite(str(out_path), canvas)
print('Saved:', out_path)
