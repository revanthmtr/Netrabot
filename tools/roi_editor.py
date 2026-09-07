#!/usr/bin/env python3
"""
ROI Verification & Editor Tool for NETRABOT Visual Inspection.

Overlays every region-of-interest (ROI) box defined in part_spec.json on a
golden/master reference image with distinct colors, coordinate labels, and dimensions.

Usage:
    # CLI overlay generation:
    python tools/roi_editor.py --image <image_path> --spec specs/part_spec.json --out output/roi_verification.png

    # Interactive GUI with live (X, Y) cursor readout:
    python tools/roi_editor.py --image <image_path> --spec specs/part_spec.json --gui
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

# Distinct colors (BGR) for ROI visualization
PALETTE = [
    (0, 140, 255),   # Orange / Amber
    (255, 200, 0),   # Cyan / Sky
    (255, 0, 180),   # Magenta / Pink
    (0, 255, 128),   # Spring Green
    (0, 220, 255),   # Yellow
    (180, 105, 255), # Hot Pink
    (255, 144, 30),  # Dodger Blue
    (50, 205, 50),   # Lime Green
]


def load_spec(spec_path: Path) -> dict:
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    with open(spec_path, "r") as f:
        return json.load(f)


def overlay_rois(
    image: np.ndarray,
    regions: List[dict],
    mm_per_px: float = 1.0,
    alpha: float = 0.18,
) -> np.ndarray:
    """
    Draw labeled ROI boxes on top of the image.
    Sorts regions by box area descending so smaller child ROIs draw on top of parents.
    """
    canvas = image.copy()
    overlay = image.copy()
    h_img, w_img = image.shape[:2]

    # Sort large boxes first so smaller nested boxes (e.g. icons inside panel) render on top
    def box_area(r):
        b = r.get("box", [0, 0, 0, 0])
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    sorted_regions = sorted(regions, key=box_area, reverse=True)

    # 1. Draw translucent fills
    for i, region in enumerate(sorted_regions):
        box = region.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = PALETTE[i % len(PALETTE)]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # 2. Draw borders, corner anchors, and text labels
    for i, region in enumerate(sorted_regions):
        box = region.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        name = region.get("name", f"ROI_{i+1}")
        color = PALETTE[i % len(PALETTE)]
        w_px = x2 - x1
        h_px = y2 - y1
        w_mm = w_px * mm_per_px
        h_mm = h_px * mm_per_px

        # Solid bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Corner crosshairs/brackets
        bracket_len = min(12, max(4, min(w_px, h_px) // 4))
        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + bracket_len, y1), color, 3)
        cv2.line(canvas, (x1, y1), (x1, y1 + bracket_len), color, 3)
        # Top-right
        cv2.line(canvas, (x2, y1), (x2 - bracket_len, y1), color, 3)
        cv2.line(canvas, (x2, y1), (x2, y1 + bracket_len), color, 3)
        # Bottom-left
        cv2.line(canvas, (x1, y2), (x1 + bracket_len, y2), color, 3)
        cv2.line(canvas, (x1, y2), (x1, y2 - bracket_len), color, 3)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - bracket_len, y2), color, 3)
        cv2.line(canvas, (x2, y2), (x2, y2 - bracket_len), color, 3)

        # Label text
        label_title = f"{name}"
        label_coords = f"[{x1},{y1},{x2},{y2}] {w_px}x{h_px}px ({w_mm:.1f}x{h_mm:.1f}mm)"

        # Text sizing
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw1, th1), b1 = cv2.getTextSize(label_title, font, font_scale, thickness)
        (tw2, th2), b2 = cv2.getTextSize(label_coords, font, font_scale - 0.06, thickness)
        label_w = max(tw1, tw2) + 12
        label_h = th1 + th2 + 10

        # Position label badge (above box if space, otherwise inside top)
        ly1 = y1 - label_h - 4 if y1 - label_h - 4 > 5 else y1 + 4
        ly2 = ly1 + label_h
        lx1 = max(4, min(x1, w_img - label_w - 4))
        lx2 = lx1 + label_w

        # Draw dark badge background with color accent bar
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (20, 20, 22), -1)
        cv2.rectangle(canvas, (lx1, ly1), (lx1 + 3, ly2), color, -1)
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (70, 70, 75), 1)

        # Draw text inside badge
        cv2.putText(
            canvas,
            label_title,
            (lx1 + 7, ly1 + th1 + 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label_coords,
            (lx1 + 7, ly1 + th1 + th2 + 7),
            font,
            font_scale - 0.06,
            (200, 200, 200),
            thickness,
            cv2.LINE_AA,
        )

    # 3. Add top-left header bar
    header_h = 42
    header = np.zeros((header_h, canvas.shape[1], 3), dtype=np.uint8)
    header[:] = (26, 26, 30)
    cv2.putText(
        header,
        f"NETRABOT ROI Verification | mm_per_px={mm_per_px:.4f} | Total Regions: {len(regions)}",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(header, (0, header_h - 1), (canvas.shape[1], header_h - 1), (60, 60, 65), 1)

    result = np.vstack([header, canvas])
    return result


def print_roi_table(regions: List[dict], mm_per_px: float, img_shape: Tuple[int, int]):
    h_img, w_img = img_shape
    print("\n" + "=" * 90)
    print(f"ROI SPECIFICATION TABLE (Image: {w_img}x{h_img} px | Scale: {mm_per_px:.4f} mm/px)")
    print("=" * 90)
    print(
        f"{'NAME':<16} {'X1':>5} {'Y1':>5} {'X2':>5} {'Y2':>5} | "
        f"{'W (px)':>7} {'H (px)':>7} | {'W (mm)':>7} {'H (mm)':>7} | {'BOUNDS':<8}"
    )
    print("-" * 90)

    for r in regions:
        name = r.get("name", "unnamed")
        box = r.get("box", [0, 0, 0, 0])
        if len(box) == 4:
            x1, y1, x2, y2 = box
            w_px = x2 - x1
            h_px = y2 - y1
            w_mm = w_px * mm_per_px
            h_mm = h_px * mm_per_px
            in_bounds = (0 <= x1 < x2 <= w_img) and (0 <= y1 < y2 <= h_img)
            bounds_str = "OK" if in_bounds else "EXCEEDS"
            print(
                f"{name:<16} {x1:>5} {y1:>5} {x2:>5} {y2:>5} | "
                f"{w_px:>7} {h_px:>7} | {w_mm:>7.2f} {h_mm:>7.2f} | {bounds_str:<8}"
            )
        else:
            print(f"{name:<16} INVALID BOX: {box}")
    print("=" * 90 + "\n")


def run_interactive_gui(annotated_img: np.ndarray, raw_img: np.ndarray, window_name: str = "NETRABOT ROI Editor"):
    """
    OpenCV GUI window with live coordinate readout on mouse move.
    """
    display = annotated_img.copy()
    status_text = "Move mouse over image to inspect (X, Y). Press 's' to save, 'q' or ESC to exit."

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        display = annotated_img.copy()
        # Offset for the 42px header added in overlay_rois
        actual_y = y - 42
        if 0 <= actual_y < raw_img.shape[0] and 0 <= x < raw_img.shape[1]:
            coord_str = f"Pixel: X={x}, Y={actual_y} | BGR={raw_img[actual_y, x].tolist()}"
        else:
            coord_str = "Outside image area"

        # Draw mouse crosshair
        if actual_y >= 0:
            cv2.line(display, (x, 42), (x, display.shape[0]), (100, 100, 100), 1)
            cv2.line(display, (0, y), (display.shape[1], y), (100, 100, 100), 1)

        # Status footer bar
        footer_y = display.shape[0] - 25
        cv2.rectangle(display, (0, footer_y - 5), (display.shape[1], display.shape[0]), (15, 15, 18), -1)
        cv2.putText(display, f"{coord_str}  |  {status_text}", (12, display.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.imshow(window_name, display)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)
        cv2.imshow(window_name, display)
        print("GUI Window opened. Hover mouse for coordinates. Press 'q' or ESC to close.")
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('s'):
                out_name = "roi_interactive_saved.png"
                cv2.imwrite(out_name, annotated_img)
                print(f"Saved current overlay to {out_name}")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Notice: GUI mode could not open a display ({e}). Output saved to file.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay part_spec.json region boxes on golden image for verification."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to golden reference image (default: first image in data/golden or data/models/*/golden).",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="specs/part_spec.json",
        help="Path to part spec JSON (default: specs/part_spec.json).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/roi_verification.png",
        help="Path to save annotated overlay image (default: output/roi_verification.png).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch interactive OpenCV inspection window with live (X, Y) coordinate cursor.",
    )
    args = parser.parse_args()

    # 1. Resolve Spec
    spec_path = Path(args.spec)
    try:
        spec = load_spec(spec_path)
    except Exception as e:
        print(f"Error loading spec: {e}", file=sys.stderr)
        sys.exit(1)

    regions = spec.get("regions", [])
    mm_per_px = float(spec.get("calibration", {}).get("mm_per_px", 1.0))

    # 2. Resolve Image
    img_path = None
    if args.image:
        img_path = Path(args.image)
    else:
        # Auto-detect from potential golden folders
        candidates = [
            Path("data/golden"),
            Path("specs"),
            Path("data"),
        ] + list(Path("data/models").glob("*/golden"))
        for c in candidates:
            if c.is_dir():
                imgs = sorted([p for p in c.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.heic', '.tif', '.bmp')])
                if imgs:
                    img_path = imgs[0]
                    break

    if not img_path or not img_path.exists():
        print("Error: No reference image found. Please provide an image using --image <path>.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading reference image from: {img_path}")
    from engine.io_utils import load_image
    try:
        image = load_image(img_path)
    except Exception as e:
        print(f"Failed to load image: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Print Table
    print_roi_table(regions, mm_per_px, image.shape[:2])

    # 4. Generate Overlay
    annotated = overlay_rois(image, regions, mm_per_px=mm_per_px)

    # 5. Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved ROI verification overlay to: {out_path}")

    # Also save to artifacts directory if it exists
    artifact_dir = Path("/Users/puttaparthirevanthsai/.gemini/antigravity-ide/brain/c3e8d1cc-3f85-4cb3-971e-b128be2f6239")
    if artifact_dir.exists():
        cv2.imwrite(str(artifact_dir / "roi_verification.png"), annotated)

    # 6. Interactive GUI if requested
    if args.gui:
        run_interactive_gui(annotated, image)


if __name__ == "__main__":
    main()
