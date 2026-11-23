"""
F.O.R.G.E. — Master Forensic Panel Generator.
Stitches controlled diagnostic shaders and before/after close-ups into high-res master review panels.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_shader_panel(diag_dir: Path, out_panel_path: Path, title: str):
    shaders = [
        ("shader_1_flat_grey_flat_shading.png", "1. Flat Grey Material (Flat Shading)"),
        ("shader_2_flat_grey_smooth_normals.png", "2. Flat Grey Material (Smooth Normals)"),
        ("shader_3_uv_checkerboard.png", "3. UV Checkerboard Texture Grid"),
        ("shader_4_actual_photographic_texture.png", "4. Actual Photographic Texture Atlas"),
        ("shader_5_normal_vector_visualization.png", "5. Normal-Vector RGB Visualization (nx, ny, nz)"),
        ("shader_6_wireframe_overlay.png", "6. Wireframe Overlay & Topology Inspection")
    ]
    
    # 2 rows x 3 columns
    card_w, card_h = 720, 450
    header_h = 70
    canvas_w = card_w * 3
    canvas_h = card_h * 2 + header_h
    
    panel = Image.new("RGB", (canvas_w, canvas_h), color=(15, 18, 24))
    draw = ImageDraw.Draw(panel)
    
    # Header
    draw.rectangle([0, 0, canvas_w, header_h], fill=(22, 27, 34))
    draw.text((30, 20), f"F.O.R.G.E. FORENSIC INVESTIGATION — {title}", fill=(56, 189, 248))
    draw.text((canvas_w - 550, 26), "Controlled Viewpoint | 1:1 Camera Pose Comparison", fill=(148, 163, 184))
    
    for idx, (fname, label) in enumerate(shaders):
        img_path = diag_dir / fname
        if not img_path.exists():
            continue
        im = Image.open(img_path).resize((card_w, card_h), Image.Resampling.LANCZOS)
        
        c = idx % 3
        r = idx // 3
        x = c * card_w
        y = header_h + r * card_h
        
        panel.paste(im, (x, y))
        
        # Overlay label bar
        draw_im = ImageDraw.Draw(panel)
        draw_im.rectangle([x, y, x + card_w, y + 36], fill=(15, 23, 42))
        draw_im.text((x + 15, y + 10), label, fill=(255, 255, 255))
        draw_im.rectangle([x, y, x + card_w, y + card_h], outline=(51, 65, 85), width=2)
        
    panel.save(out_panel_path, quality=95)
    print(f"[+] Saved Master Shader Diagnostic Panel: {out_panel_path.name}")


def build_before_after_panel(diag_dir: Path, out_panel_path: Path):
    pairs = [
        ("comparison_before_stage1_furniture_closeup.png", "comparison_after_stage2_furniture_closeup.png", "A. Furniture Close-Up (Student Desk & Monitor)"),
        ("comparison_before_stage1_wall_closeup.png", "comparison_after_stage2_wall_closeup.png", "B. Interior Wall Close-Up (Plaster & Edge)"),
        ("comparison_before_stage1_floor_closeup.png", "comparison_after_stage2_floor_closeup.png", "C. Room Floor Close-Up (Tile Grid & Perspective)")
    ]
    
    col_w, col_h = 720, 450
    header_h = 70
    canvas_w = col_w * 2
    canvas_h = col_h * 3 + header_h
    
    panel = Image.new("RGB", (canvas_w, canvas_h), color=(15, 18, 24))
    draw = ImageDraw.Draw(panel)
    
    # Header
    draw.rectangle([0, 0, canvas_w, header_h], fill=(22, 27, 34))
    draw.text((30, 20), "F.O.R.G.E. VISUAL QUALITY AUDIT — STAGE 1 (BEFORE) VS STAGE 2 (AFTER)", fill=(56, 189, 248))
    draw.text((canvas_w - 480, 26), "Identical Perspective Close-Up Verification", fill=(148, 163, 184))
    
    for row_idx, (b_name, a_name, row_label) in enumerate(pairs):
        b_path = diag_dir / b_name
        a_path = diag_dir / a_name
        
        y = header_h + row_idx * col_h
        
        if b_path.exists():
            im_b = Image.open(b_path).resize((col_w, col_h), Image.Resampling.LANCZOS)
            panel.paste(im_b, (0, y))
            draw.rectangle([0, y, col_w, y + 36], fill=(15, 23, 42))
            draw.text((15, y + 10), f"STAGE 1 (Greedy Argmax): {row_label}", fill=(248, 113, 113))
            draw.rectangle([0, y, col_w, y + col_h], outline=(51, 65, 85), width=2)
            
        if a_path.exists():
            im_a = Image.open(a_path).resize((col_w, col_h), Image.Resampling.LANCZOS)
            panel.paste(im_a, (col_w, y))
            draw.rectangle([col_w, y, col_w * 2, y + 36], fill=(15, 23, 42))
            draw.text((col_w + 15, y + 10), f"STAGE 2 (MRF + Crease Normals): {row_label}", fill=(74, 222, 128))
            draw.rectangle([col_w, y, col_w * 2, y + col_h], outline=(51, 65, 85), width=2)
            
    panel.save(out_panel_path, quality=95)
    print(f"[+] Saved Before/After Comparison Panel: {out_panel_path.name}")


def main():
    dir_v1 = PROJECT_ROOT / "outputs/indoor/stage2_texture_video1/diagnostic_renders"
    out_shader_v1 = dir_v1 / "master_shader_diagnostic_review.png"
    out_ba_v1 = dir_v1 / "master_closeups_before_after.png"
    
    build_shader_panel(dir_v1, out_shader_v1, "Student Hostel Room B 314")
    build_before_after_panel(dir_v1, out_ba_v1)
    
    dir_v2 = PROJECT_ROOT / "outputs/indoor/stage2_texture_video2/diagnostic_renders"
    out_shader_v2 = dir_v2 / "master_shader_diagnostic_review.png"
    build_shader_panel(dir_v2, out_shader_v2, "Hostel Corridor Walkthrough")

if __name__ == "__main__":
    main()

