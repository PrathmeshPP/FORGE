import cv2, numpy as np
from pathlib import Path

v11_dir = Path('C:/Users/25beevdt025/Desktop/FORGE/outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v11')

W, H = 2000, 1350
canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

# Header
cv2.putText(canvas, 'F.O.R.G.E V11 - Neutral Geometry & Surface Quality Comparative Audit', (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.05, (15, 15, 15), 2, cv2.LINE_AA)
cv2.putText(canvas, 'Evaluating Surface Continuity, Planarity, and Topological Integrity across Candidates C (V9), Flawed V10, and Candidate A Pilot', (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 90, 90), 1, cv2.LINE_AA)
cv2.line(canvas, (40, 130), (1960, 130), (220, 220, 220), 1)

# Load renders
im_v9 = cv2.imread(str(v11_dir / 'v9_render_oblique.png'))
im_v10 = cv2.imread(str(v11_dir / 'v10_render_oblique.png'))
im_pilot = cv2.imread(str(v11_dir / 'test_mesh_continuous_oblique.png'))

panels = [
    {
        'title': 'Candidate C - V9 Reference Mesh',
        'sub': 'Monocular Relative Depth Fusion Mesh',
        'img': im_v9,
        'status': 'RETAINED VISUAL BASELINE',
        'status_color': (0, 140, 0),
        'metrics': [
            'Triangle Count: 1,386,245 faces (695k vertices)',
            'Planarity: Smooth, coherent planar roof & road',
            'Continuity: Recognizable building & terrain structure',
            'Defects: Squashed elevation (0.58m), non-manifold edges',
            'Verdict: Preserved as existing visual baseline'
        ],
        'x': 50
    },
    {
        'title': 'Candidate A Flawed - V10 Disparity Mesh',
        'sub': 'Hardcoded Range Truncation (numDisp = 160)',
        'img': im_v10,
        'status': 'FATAL SHATTERED GEOMETRY',
        'status_color': (0, 0, 180),
        'metrics': [
            'Triangle Count: 1,364,145 faces (683k vertices)',
            'Structural Void: Missing building roof & courtyard',
            'Mechanism: Disparity < 160 cut off foreground points',
            'Downstream: Poisson low-density bubble hollowed out',
            'Decimation: Quadric collapse created sharp jagged shards'
        ],
        'x': 690
    },
    {
        'title': 'Candidate A Corrected - V11 Pilot Mesh',
        'sub': 'Adaptive Disparity Range [0, 512] (Frames 5-11)',
        'img': im_pilot,
        'status': 'GEOMETRY FAILED PLANARITY REVIEW',
        'status_color': (0, 100, 200),
        'metrics': [
            'Triangle Count: 902,289 faces (462k vertices)',
            'Structural Recovery: 100% continuous roof & courtyard',
            'Spatial Bounds: True 3D elevation profile restored',
            'Artifact: SGBM 1D quantization causes +-0.15 unit ripples',
            'Verdict: Crumpled foil texture fails architectural review'
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
cv2.putText(canvas, 'MANDATORY SECTION 9 GATEKEEPING & DECISION', (60, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 50, 130), 2, cv2.LINE_AA)
cv2.putText(canvas, '1. Section 9 Mandate: Do not create another final textured model until the neutral mesh passes the geometry review.', (60, by + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
cv2.putText(canvas, '2. Candidate A Failure: While topological coverage is restored, 1D SGBM disparity quantization noise creates an unacceptable bumpy surface on flat architecture.', (60, by + 87), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
cv2.putText(canvas, '3. Decision: Texture baking is WITHHELD. V9 remains the active visual baseline. Candidate A proves the root cause but cannot be textured into production.', (60, by + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)

out_path = v11_dir / 'neutral_geometry_preview.png'
cv2.imwrite(str(out_path), canvas)
print('Saved:', out_path)
