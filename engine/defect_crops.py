"""
Defect evidence crops: MASTER vs SAMPLE side-by-side.

For each detected defect, produce a zoomed comparison showing the same
region on the golden reference and on the sample, with a difference
heatmap. This is what an operator actually needs to confirm or reject a
detection -- a box on a full-size image is not enough to judge a 3mm2
speck.

Output per defect:
    <out>/crops/<sample>_defect_<id>.png
Layout:
    [ MASTER ]  [ SAMPLE ]  [ DIFF HEATMAP ]
"""

import cv2
import numpy as np
import os


def _label_bar(width, text, height=26, bg=(32, 32, 32), fg=(255, 255, 255)):
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, fg, 1, cv2.LINE_AA)
    return bar


def make_defect_crop(golden_bgr, sample_bgr, defect, out_path,
                     context_px=60, zoom=4, mm_per_px=1.0):
    """
    Build a side-by-side evidence image for one defect.

    golden_bgr : golden reference image (BGR)
    sample_bgr : REGISTERED sample image (BGR) - must be aligned already
    defect     : dict with x, y, w, h, id, severity, area_mm2, defect_type
    """
    H, W = golden_bgr.shape[:2]
    x, y = int(defect["x"]), int(defect["y"])
    w, h = max(1, int(defect["w"])), max(1, int(defect["h"]))

    # Context window around the defect, clamped to image bounds
    x1 = max(0, x - context_px)
    y1 = max(0, y - context_px)
    x2 = min(W, x + w + context_px)
    y2 = min(H, y + h + context_px)
    if x2 <= x1 or y2 <= y1:
        return None

    g = golden_bgr[y1:y2, x1:x2].copy()
    s = sample_bgr[y1:y2, x1:x2].copy()
    if g.size == 0 or s.size == 0 or g.shape != s.shape:
        return None

    # Difference heatmap over the same window
    gg = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    d = cv2.absdiff(gg, sg)
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(d, cv2.COLORMAP_JET)

    # Mark the defect box (coords relative to the crop) on sample + heat
    rx, ry = x - x1, y - y1
    for img in (s, heat):
        cv2.rectangle(img, (rx - 2, ry - 2), (rx + w + 2, ry + h + 2),
                      (0, 0, 255), 1)

    # Zoom all three panels
    interp = cv2.INTER_NEAREST if zoom >= 3 else cv2.INTER_LINEAR
    g = cv2.resize(g, None, fx=zoom, fy=zoom, interpolation=interp)
    s = cv2.resize(s, None, fx=zoom, fy=zoom, interpolation=interp)
    heat = cv2.resize(heat, None, fx=zoom, fy=zoom, interpolation=interp)

    pw = g.shape[1]
    panels = [
        np.vstack([_label_bar(pw, "MASTER (golden)"), g]),
        np.vstack([_label_bar(pw, "SAMPLE (defect)", bg=(0, 0, 120)), s]),
        np.vstack([_label_bar(pw, "DIFFERENCE"), heat]),
    ]
    sep = np.full((panels[0].shape[0], 4, 3), 90, dtype=np.uint8)
    combo = np.hstack([panels[0], sep, panels[1], sep, panels[2]])

    # Header with the defect facts
    sev = defect.get("severity", "?")
    hdr_bg = {"CRITICAL": (0, 0, 140), "MAJOR": (0, 90, 160),
              "MINOR": (0, 130, 140)}.get(sev, (60, 60, 60))
    hdr = np.full((34, combo.shape[1], 3), hdr_bg, dtype=np.uint8)
    txt = (f"#{defect.get('id')}  {sev}  "
           f"{defect.get('area_mm2', 0):.3f} mm2  "
           f"({defect.get('w')}x{defect.get('h')} px)  "
           f"at ({x},{y})  {str(defect.get('defect_type',''))[:42]}")
    cv2.putText(hdr, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    combo = np.vstack([hdr, combo])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, combo)
    return out_path


def generate_all_crops(golden_bgr, sample_bgr, defects, out_dir,
                       sample_name, mm_per_px=1.0, max_crops=25):
    """Generate evidence crops for the top-ranked defects."""
    paths = []
    for d in defects[:max_crops]:
        if not d.get("w"):          # skip global/topology defects
            continue
        # Scale zoom to defect size so small specks are actually visible
        span = max(d["w"], d["h"])
        zoom = 8 if span < 12 else (5 if span < 40 else 3)
        p = os.path.join(out_dir, "crops",
                         f"{sample_name}_defect_{d.get('id')}.png")
        r = make_defect_crop(golden_bgr, sample_bgr, d, p,
                             zoom=zoom, mm_per_px=mm_per_px)
        if r:
            d["crop_url"] = os.path.relpath(r, out_dir)
            paths.append(r)
    return paths
