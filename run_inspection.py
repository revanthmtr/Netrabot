#!/usr/bin/env python3
"""
AOI Inspection CLI Runner.

Usage:
    python run_inspection.py --golden data/golden --input data/samples --spec specs/part_spec.json --out output --top-n 15
    python run_inspection.py --input data/samples --consensus --spec specs/part_spec.json --out output --top-n 15
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from skimage.color import rgb2lab

from engine.calibrated_engine import CalibratedEngine
from engine.inspection_engine import (
    Defect,
    GoldenReference,
    InspectionEngine,
    InspectionResult,
    report,
)
from engine.io_utils import build_part_mask, load_image

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".heic",
    ".heif",
    ".bmp",
    ".tiff",
    ".tif",
}


def find_image_files(path: Path) -> List[Path]:
    """Find all supported image files in a path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            return [path]
        return []
    elif path.is_dir():
        files = [
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        return sorted(files)
    return []


def annotate_defects(img: np.ndarray, defects: List[Defect]) -> np.ndarray:
    """Annotate image with bounding boxes, labels, and sample counts."""
    vis = img.copy()
    colors = {
        "CRITICAL": (0, 0, 255),
        "MAJOR": (0, 140, 255),
        "MINOR": (0, 220, 255),
    }
    for d in defects:
        if d.w == 0 or d.h == 0:
            continue
        c = colors.get(d.severity, (255, 255, 255))
        pad = 6
        x1 = max(0, d.x - pad)
        y1 = max(0, d.y - pad)
        x2 = min(vis.shape[1] - 1, d.x + d.w + pad)
        y2 = min(vis.shape[0] - 1, d.y + d.h + pad)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)

        sample_tag = f"[{d.__dict__['sample_str']}] " if "sample_str" in d.__dict__ else ""
        lbl = f"#{d.id} {sample_tag}{d.severity[:4]} {d.area_mm2:.2f}mm2"
        ly = max(14, y1 - 5)
        cv2.putText(
            vis, lbl, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            vis, lbl, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA
        )
    return vis


def format_ranked_report(
    res: InspectionResult, name: str = "", top_n: Optional[int] = None
) -> str:
    """Format inspection report, displaying up to top_n ranked defects."""
    L = []
    L.append("=" * 72)
    L.append(f"INSPECTION REPORT{(' - ' + name) if name else ''}")
    L.append("=" * 72)
    L.append(f"VERDICT: {res.verdict}")
    if res.gate_failures:
        L.append("\
GATE / GLOBAL FAILURES:")
        for g in res.gate_failures:
            L.append(f"  ! {g}")

    consensus = res.global_metrics.get("consensus")
    if consensus:
        L.append("\
CONSENSUS AUDIT FILTERING:")
        L.append(
            f"  Total raw candidates detected:      {consensus['total_raw_candidates']}"
        )
        L.append(
            f"  Suppressed systematic noise (>60%): {consensus['suppressed_count']}"
        )
        L.append(
            f"  Surviving real defect candidates:   {consensus['surviving_count']}"
        )
        L.append(
            f"  Sample occurrence threshold:        <= {consensus['threshold_pct']:.0f}% of {consensus['n_samples']} samples"
        )

    reg = res.global_metrics.get("registration")
    if reg:
        L.append(
            f"\
Registration: shift {reg['shift_px']:.2f}px "
            f"({reg['shift_mm']:.3f}mm)  conf={reg['response']:.3f}"
        )
    topo = res.global_metrics.get("topology")
    if topo:
        L.append(
            f"Topology: {topo['contour_count']} elements "
            f"(golden {topo['golden_mean']:.1f} +/- tol {topo['tolerance']:.1f})"
            f"{'  <-- DEVIATED' if topo['deviated'] else ''}"
        )

    if res.detector_stats:
        L.append("\
DETECTOR ACTIVITY:")
        for k, v in res.detector_stats.items():
            L.append(f"  {k:10s} {v['flagged_px']:>8d} px  ({v['pct']:.4f}%)")

    total_candidates = len(res.defects)
    shown = min(top_n, total_candidates) if top_n is not None else total_candidates
    L.append(f"\
DEFECTS FOUND: {total_candidates}")
    if total_candidates > 0:
        if top_n is not None and top_n < total_candidates:
            L.append(
                f"(Showing top {shown} of {total_candidates} candidates ranked by saliency score)"
            )
        has_samples = any("sample_str" in d.__dict__ for d in res.defects[:shown])
        if has_samples:
            L.append(
                f"{'RNK':>3} {'ID':>3} {'SEV':<8} {'AREA mm2':>8} {'SALIENCY':>8} "
                f"{'POS':<12} {'SAMPLES':<8} {'TYPE':<28} DETECTORS"
            )
            L.append("-" * 84)
            for i, d in enumerate(res.defects[:shown]):
                pos = f"({d.x},{d.y})" if d.w else "global"
                saliency = d.__dict__.get("saliency", 0.0)
                rank = d.__dict__.get("rank", i + 1)
                sample_str = d.__dict__.get("sample_str", "-")
                det_str = ",".join(d.detectors)
                dtype = (
                    (d.defect_type[:25] + "...")
                    if len(d.defect_type) > 28
                    else d.defect_type
                )
                L.append(
                    f"{rank:>3} {d.id:>3} {d.severity:<8} {d.area_mm2:>8.3f} "
                    f"{saliency:>8.4f} {pos:<12} {sample_str:<8} {dtype:<28} {det_str}"
                )
                tm = InspectionEngine.top_measurement(getattr(d, "measurements", None))
                if tm:
                    L.append(f"      -> MEASURED: {tm}")
        else:
            L.append(
                f"{'RNK':>3} {'ID':>3} {'SEV':<8} {'AREA mm2':>8} {'SALIENCY':>8} "
                f"{'POS':<12} {'TYPE':<32} DETECTORS"
            )
            L.append("-" * 78)
            for i, d in enumerate(res.defects[:shown]):
                pos = f"({d.x},{d.y})" if d.w else "global"
                saliency = d.__dict__.get("saliency", 0.0)
                rank = d.__dict__.get("rank", i + 1)
                det_str = ",".join(d.detectors)
                dtype = (
                    (d.defect_type[:29] + "...")
                    if len(d.defect_type) > 32
                    else d.defect_type
                )
                L.append(
                    f"{rank:>3} {d.id:>3} {d.severity:<8} {d.area_mm2:>8.3f} "
                    f"{saliency:>8.4f} {pos:<12} {dtype:<32} {det_str}"
                )
                tm = InspectionEngine.top_measurement(getattr(d, "measurements", None))
                if tm:
                    L.append(f"      -> MEASURED: {tm}")
    L.append("=" * 72)
    return "\n".join(L)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AOI Visual Inspection on printed safety labels."
    )
    parser.add_argument(
        "--golden",
        type=str,
        default="data/golden",
        help="Folder containing known-good golden reference images.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Single image file or folder of images to inspect.",
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
        default="output",
        help="Output base folder (default: output).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of ranked defects to display and annotate (default: 15).",
    )
    parser.add_argument(
        "--consensus",
        action="store_true",
        help="Enable consensus mode: build golden reference from median of input samples and suppress systematic noise appearing in >60% of samples.",
    )
    return parser.parse_args()


def run_consensus_inspection(
    input_files: List[Path],
    spec: Dict,
    args: argparse.Namespace,
    annotated_dir: Path,
    reports_dir: Path,
):
    """
    Consensus Mode:
    Given N sample images of the same part type:
    1. Build statistical golden reference from per-pixel MEDIAN across all samples.
    2. Inspect each sample against the median reference.
    3. Calculate cross-sample occurrence frequency for each candidate.
    4. Suppress candidates appearing in >60% of samples as systematic noise.
    5. Retain candidates appearing in <=60% of samples as real defects.
    6. Maintain a complete audit log of all suppressed and surviving candidates.
    """
    n_samples = len(input_files)
    if n_samples < 2:
        print(
            f"Error: Consensus mode requires at least 2 sample images (received {n_samples}).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Running CONSENSUS MODE across {n_samples} samples...\
"
        f"  -> Step 1: Loading all {n_samples} sample images..."
    )
    sample_images = []
    for f in input_files:
        try:
            im = load_image(f)
            sample_images.append(im)
        except Exception as e:
            print(f"Error loading {f}: {e}", file=sys.stderr)
            sys.exit(1)

    mm_per_px = float(spec.get("calibration", {}).get("mm_per_px", 1.0))
    sensitivity = float(spec.get("detection", {}).get("sensitivity", 0.85))
    min_reg_conf = float(spec.get("detection", {}).get("min_registration_confidence", 0.6))
    regions = spec.get("regions", [])
    roi_map = {r["name"]: r["box"] for r in regions if "name" in r and "box" in r}

    # Build median reference
    print("  -> Step 2: Building statistical golden reference from per-pixel MEDIAN...")
    part_mask = build_part_mask(sample_images[0])
    golden_ref = GoldenReference(mm_per_px=mm_per_px)
    golden_ref.build_from_median(sample_images, part_mask=part_mask)
    engine = CalibratedEngine(golden_ref, sensitivity=sensitivity, roi_map=roi_map,
                               min_registration_confidence=min_reg_conf)

    # Stage 1: Feature extraction & registration per sample
    print("  -> Step 3: Running multi-detector fusion across all samples...")
    sample_data = []
    for f, img in zip(input_files, sample_images):
        s_reg, reg_info = engine.register(img)
        gray = cv2.cvtColor(s_reg, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        gref = golden_ref.golden_gray

        masks = {}
        sigs = {}
        masks["zscore"], sigs["zscore"] = engine._d1_zscore(gray_f)
        masks["ssim"], sigs["ssim"] = engine._d2_multiscale_ssim(gray, gref)
        masks["absdiff"], sigs["absdiff"] = engine._d3_absdiff(gray, gref)
        masks["color"], sigs["color"] = engine._d4_color_lab(s_reg)
        masks["edge"], sigs["edge"] = engine._d5_edge_topology(gray)
        masks["morph"], sigs["morph"] = engine._d6_morphological(gray, gref)
        masks["gradient"], sigs["gradient"] = engine._d7_gradient(gray, gref)
        masks["texture"], sigs["texture"] = engine._d8_texture(gray, gref)

        u = np.zeros(gray.shape, dtype=np.uint8)
        for m in masks.values():
            u = cv2.bitwise_or(u, m)
        u = cv2.bitwise_and(u, u, mask=golden_ref.part_mask)

        cnt, topo_dev, topo_tol = engine._d9_topology(gray)

        sample_data.append({
            "file": f,
            "img": img,
            "registered": s_reg,
            "gray": gray,
            "masks": masks,
            "sigs": sigs,
            "union": u,
            "reg_info": reg_info,
            "contour_count": cnt,
            "topo_dev": topo_dev,
            "topo_tol": topo_tol,
        })

    # Step 4: Compute systematic noise frequency mask across samples
    print("  -> Step 4: Measuring cross-sample consensus noise floor (>60% threshold)...")
    stack = np.stack([(sd["union"] > 0).astype(np.uint8) for sd in sample_data])
    freq_map = stack.sum(axis=0)
    systematic_noise_mask = (freq_map / n_samples) > 0.60
    noise_px = int(systematic_noise_mask.sum())
    print(f"     Isolated {noise_px} systematic noise pixels appearing across >60% of samples.")

    # Step 5: Check regional ΔE across samples
    region_checks = []
    gh, gw = golden_ref.golden_bgr.shape[:2]
    for r in regions:
        r_name = r.get("name")
        box = r.get("box")
        tol_dE = r.get("color_tolerance_dE")
        if box and tol_dE:
            x1 = max(0, min(gw, box[0]))
            y1 = max(0, min(gh, box[1]))
            x2 = max(0, min(gw, box[2]))
            y2 = max(0, min(gh, box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            g_crop = golden_ref.golden_bgr[y1:y2, x1:x2]
            if g_crop.size == 0:
                continue
            g_lab = rgb2lab(cv2.cvtColor(g_crop, cv2.COLOR_BGR2RGB) / 255.0)
            des = []
            for sd in sample_data:
                s_crop = sd["registered"][y1:y2, x1:x2]
                if s_crop.size == 0 or s_crop.shape != g_crop.shape:
                    continue
                s_lab = rgb2lab(cv2.cvtColor(s_crop, cv2.COLOR_BGR2RGB) / 255.0)
                dE = float(np.linalg.norm(s_lab - g_lab, axis=-1).mean())
                des.append(dE)
            if len(des) == len(sample_data):
                region_checks.append({
                    "name": r_name,
                    "box": [x1, y1, x2, y2],
                    "tolerance": tol_dE,
                    "dEs": des,
                })

    # Step 6: Extract candidates, filter by consensus, and audit
    print("  -> Step 5: Extracting and ranking candidates per sample...\
")
    overall_audit = {
        "mode": "consensus",
        "n_samples": n_samples,
        "consensus_threshold_pct": 60.0,
        "spec": spec.get("part_id", "UNKNOWN"),
        "parts": {},
    }

    for idx, sd in enumerate(sample_data):
        file_path = sd["file"]
        # Full candidates before suppression (for audit log)
        u_full = cv2.morphologyEx(sd["union"], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n_full, lbl_full, stats_full, cents_full = cv2.connectedComponentsWithStats(u_full, 8)
        raw_candidates_count = n_full - 1

        # Clean union components after systematic noise subtraction
        u_clean = sd["union"].copy()
        u_clean[systematic_noise_mask] = 0
        u_clean = cv2.morphologyEx(u_clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        u_clean = cv2.morphologyEx(u_clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n_clean, lbl_clean, stats_clean, cents_clean = cv2.connectedComponentsWithStats(u_clean, 8)

        surviving_defects: List[Defect] = []
        suppressed_candidates: List[Dict] = []
        did = 0

        # Audit log: Record suppressed candidates from raw detection that were identified as systematic noise
        for k in range(1, n_full):
            fx, fy, fw, fh, farea = stats_full[k]
            farea_mm2 = farea * (mm_per_px ** 2)
            fcomp = (lbl_full == k)
            noise_ratio = float((systematic_noise_mask & fcomp).sum()) / max(1.0, float(farea))
            if noise_ratio > 0.30:
                suppressed_candidates.append({
                    "raw_id": k,
                    "x": int(fx),
                    "y": int(fy),
                    "w": int(fw),
                    "h": int(fh),
                    "area_mm2": round(float(farea_mm2), 3),
                    "reason": f"systematic noise (>60% sample frequency, {noise_ratio * 100:.1f}% noise overlap)",
                })

        for i in range(1, n_clean):
            x, y, w, h, area = stats_clean[i]
            area_mm2 = area * (mm_per_px ** 2)
            if area_mm2 < engine.min_defect_mm2:
                continue

            comp = (lbl_clean == i)

            # Check sample occurrence across all N samples
            hits = 1
            for j in range(n_samples):
                if j == idx:
                    continue
                overlap = ((sample_data[j]["union"] > 0) & comp).sum() / max(1, area)
                if overlap > 0.35:
                    hits += 1

            sample_ratio = hits / n_samples
            sample_str = f"{hits}/{n_samples}"

            fired = [k for k, m in sd["masks"].items() if (m[comp] > 0).any()]
            conf = len(fired) / len(sd["masks"])
            peak = 0.0
            for k in ("zscore", "ssim", "color", "absdiff"):
                if k in sd["sigs"]:
                    peak = max(peak, float(sd["sigs"][k][comp].max()))

            d = Defect(
                id=did + 1,
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area_px=float(area),
                area_mm2=float(area_mm2),
                centroid=(float(cents_clean[i][0]), float(cents_clean[i][1])),
                detectors=fired,
                confidence=round(conf, 3),
                severity=engine._severity(area_mm2, conf, fired),
                defect_type=engine._classify(fired, sd["sigs"], comp, sd["registered"]),
                peak_deviation=round(peak, 2),
                region=engine._which_region(x, y, w, h),
                measurements=engine._measurements(fired, sd["sigs"], comp, engine.T),
            )
            d.__dict__["sample_count"] = hits
            d.__dict__["sample_ratio"] = sample_ratio
            d.__dict__["sample_str"] = sample_str
            d.__dict__["saliency"] = round(CalibratedEngine.saliency(d), 4)

            # Consensus filtering: appear in <= 60% of samples -> Real defect
            if sample_ratio <= 0.60:
                did += 1
                d.id = did
                surviving_defects.append(d)
            else:
                suppressed_candidates.append({
                    "id": i,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area_mm2": float(area_mm2),
                    "detectors": fired,
                    "sample_count": hits,
                    "sample_ratio": sample_ratio,
                    "reason": f"systematic noise (present in {sample_str} samples > 60%)",
                })

        # Check topology deviation
        topo_hits = sum([1 for other in sample_data if other["topo_dev"]])
        if sd["topo_dev"] and (topo_hits / n_samples <= 0.60):
            did += 1
            d_topo = Defect(
                id=did,
                x=0,
                y=0,
                w=0,
                h=0,
                area_px=0,
                area_mm2=0,
                centroid=(0, 0),
                detectors=["topology"],
                confidence=1.0,
                severity="MAJOR",
                defect_type="TOPOLOGY_CHANGE (merged/split elements - ink bridge or break)",
                peak_deviation=abs(sd["contour_count"] - golden_ref.mean_contour_count),
                region="global",
                measurements={"topology": {
                    "label": "connected element count",
                    "unit": "elements",
                    "value": sd["contour_count"],
                    "threshold": round(golden_ref.mean_contour_count + sd["topo_tol"], 2),
                    "golden_mean": round(golden_ref.mean_contour_count, 2),
                    "exceeds_by_pct": round(100.0 * (abs(sd["contour_count"] - golden_ref.mean_contour_count) / sd["topo_tol"] - 1.0), 1) if sd["topo_tol"] else None,
                }},
            )
            d_topo.__dict__["sample_count"] = topo_hits
            d_topo.__dict__["sample_ratio"] = topo_hits / n_samples
            d_topo.__dict__["sample_str"] = f"{topo_hits}/{n_samples}"
            d_topo.__dict__["saliency"] = 0.95
            surviving_defects.append(d_topo)

        # Check regional ΔE color defect
        for rc in region_checks:
            dE_val = rc["dEs"][idx]
            if dE_val > rc["tolerance"]:
                dE_hits = sum([1 for val in rc["dEs"] if val > rc["tolerance"]])
                if (dE_hits / n_samples) <= 0.60:
                    did += 1
                    bx = rc["box"]
                    d_reg = Defect(
                        id=did,
                        x=bx[0],
                        y=bx[1],
                        w=bx[2] - bx[0],
                        h=bx[3] - bx[1],
                        area_px=float((bx[2] - bx[0]) * (bx[3] - bx[1])),
                        area_mm2=float((bx[2] - bx[0]) * (bx[3] - bx[1]) * (mm_per_px ** 2)),
                        centroid=((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2),
                        detectors=["color"],
                        confidence=1.0,
                        severity="CRITICAL",
                        defect_type=f"REGIONAL_COLOR_FADE (dE={dE_val:.2f} vs tol {rc['tolerance']})",
                        peak_deviation=round(dE_val, 2),
                        region=rc["name"],
                        measurements={"color_regional": {
                            "label": "perceptual color deviation (CIE Lab dE, region average)",
                            "unit": "dE",
                            "value": round(dE_val, 3),
                            "threshold": rc["tolerance"],
                            "exceeds_by_pct": round(100.0 * (dE_val / rc["tolerance"] - 1.0), 1) if rc["tolerance"] else None,
                        }},
                    )
                    d_reg.__dict__["sample_count"] = dE_hits
                    d_reg.__dict__["sample_ratio"] = dE_hits / n_samples
                    d_reg.__dict__["sample_str"] = f"{dE_hits}/{n_samples}"
                    d_reg.__dict__["saliency"] = 0.98
                    surviving_defects.append(d_reg)

        # Rank surviving defects by composite saliency score
        surviving_defects.sort(key=lambda d: -d.__dict__.get("saliency", 0.0))
        for i, d in enumerate(surviving_defects):
            d.__dict__["rank"] = i + 1

        suppressed_count = raw_candidates_count - len(surviving_defects)
        verdict = "FAIL" if len(surviving_defects) > 0 else "PASS"

        # Build InspectionResult object
        res = InspectionResult(verdict=verdict)
        res.defects = surviving_defects
        res.gate_failures = engine.quality_gate(sd["registered"])
        reg_response = sd["reg_info"].get("response", 1.0)
        if reg_response < engine.min_registration_confidence:
            # Same failure mode as the standard track: a badly-aligned
            # frame produces hundreds/thousands of edge-jitter candidates
            # that the consensus filter cannot reliably tell apart from a
            # real, isolated defect. Flag it so the operator knows this
            # sample's result is not trustworthy, even though (unlike the
            # standard track) we don't skip it here -- doing so would
            # distort the cross-sample systematic-noise frequency count
            # the other samples depend on.
            res.verdict = "REVIEW"
            res.gate_failures.append(
                f"REGISTRATION CONFIDENCE TOO LOW: {reg_response:.3f} "
                f"(need >= {engine.min_registration_confidence}) -- alignment "
                f"unreliable, candidates below are not trustworthy")
        res.global_metrics = {
            "registration": sd["reg_info"],
            "topology": {
                "contour_count": sd["contour_count"],
                "golden_mean": golden_ref.mean_contour_count,
                "tolerance": sd["topo_tol"],
                "deviated": bool(sd["topo_dev"]),
            },
            "consensus": {
                "total_raw_candidates": raw_candidates_count,
                "suppressed_count": suppressed_count,
                "surviving_count": len(surviving_defects),
                "threshold_pct": 60.0,
                "n_samples": n_samples,
            },
        }
        res.detector_stats = {
            k: {
                "flagged_px": int((m > 0).sum()),
                "pct": round(100.0 * (m > 0).sum() / max(1, golden_ref.part_mask.sum() / 255), 4),
            }
            for k, m in sd["masks"].items()
        }
        res.annotated_image = annotate_defects(sd["registered"], surviving_defects)

        # Print report
        print(format_ranked_report(res, name=file_path.name, top_n=args.top_n))

        # Save annotated image
        annotated_path = annotated_dir / f"{file_path.stem}_annotated.png"
        cv2.imwrite(str(annotated_path), res.annotated_image)
        print(f"  -> Saved annotated image to: {annotated_path}")

        # Generate MASTER vs SAMPLE evidence crops for each defect.
        # A box on a full-size image cannot convey a 3mm2 speck; the
        # operator needs the zoomed side-by-side to confirm or reject.
        try:
            from engine.defect_crops import generate_all_crops
            crops_root = annotated_dir.parent
            defect_dicts = [d.to_dict() for d in surviving_defects]
            crop_paths = generate_all_crops(
                golden_ref.golden_bgr,
                sd["registered"],
                defect_dicts,
                str(crops_root),
                file_path.stem,
                mm_per_px=golden_ref.mm_per_px,
            )
            # push crop_url back onto the Defect objects for the JSON report
            by_id = {dd["id"]: dd for dd in defect_dicts}
            for d in surviving_defects:
                cu = by_id.get(d.id, {}).get("crop_url")
                if cu:
                    d.__dict__["crop_url"] = cu
            if crop_paths:
                print(f"  -> Saved {len(crop_paths)} defect evidence crop(s) "
                      f"to: {crops_root}/crops/")
        except Exception as e:
            print(f"  !! Defect crop generation failed: {e}")

        # Save JSON report per part with complete audit log
        report_json = {
            "part_id": spec.get("part_id", "UNKNOWN"),
            "file": file_path.name,
            "verdict": res.verdict,
            "mode": "consensus",
            "gate_failures": res.gate_failures,
            "global_metrics": res.global_metrics,
            "detector_stats": res.detector_stats,
            "total_raw_candidates": raw_candidates_count,
            "suppressed_systematic_noise_count": suppressed_count,
            "surviving_defects_count": len(surviving_defects),
            "top_n_shown": min(args.top_n, len(surviving_defects)),
            "defects": [
                {
                    **d.to_dict(),
                    "sample_count": d.__dict__.get("sample_count", 1),
                    "sample_ratio": d.__dict__.get("sample_ratio", 1.0 / n_samples),
                    "sample_str": d.__dict__.get("sample_str", f"1/{n_samples}"),
                    "saliency": d.__dict__.get("saliency", 0.0),
                    "rank": d.__dict__.get("rank", i + 1),
                    "crop_url": d.__dict__.get("crop_url"),
                }
                for i, d in enumerate(surviving_defects)
            ],
            "suppressed_candidates": suppressed_candidates,
        }
        json_path = reports_dir / f"{file_path.stem}_report.json"
        with open(json_path, "w") as f:
            json.dump(report_json, f, indent=2)
        print(f"  -> Saved JSON report & audit log to: {json_path}\
")

        overall_audit["parts"][file_path.name] = {
            "verdict": verdict,
            "raw_candidates": raw_candidates_count,
            "suppressed_count": suppressed_count,
            "surviving_count": len(surviving_defects),
            "surviving_defects": [
                {
                    "id": d.id,
                    "severity": d.severity,
                    "area_mm2": d.area_mm2,
                    "saliency": d.__dict__.get("saliency"),
                    "pos": f"({d.x},{d.y})" if d.w else "global",
                    "samples": d.__dict__.get("sample_str"),
                    "type": d.defect_type,
                    "region": d.region,
                }
                for d in surviving_defects
            ],
        }

    # Save master consensus audit file
    audit_file = reports_dir / "consensus_audit.json"
    with open(audit_file, "w") as f:
        json.dump(overall_audit, f, indent=2)
    print(f"[Audit Complete] Master consensus audit log saved to: {audit_file}")


def run_standard_inspection(
    input_files: List[Path],
    spec: Dict,
    args: argparse.Namespace,
    annotated_dir: Path,
    reports_dir: Path,
):
    """Standard single-image or multi-golden reference inspection."""
    mm_per_px = float(spec.get("calibration", {}).get("mm_per_px", 1.0))
    sensitivity = float(spec.get("detection", {}).get("sensitivity", 0.85))
    min_reg_conf = float(spec.get("detection", {}).get("min_registration_confidence", 0.6))
    noise_percentile = float(
        spec.get("detection", {}).get("noise_floor_percentile", 99.5)
    )
    regions = spec.get("regions", [])
    roi_map = {r["name"]: r["box"] for r in regions if "name" in r and "box" in r}

    # Load Golden Reference Images
    golden_dir = Path(args.golden)
    golden_files = find_image_files(golden_dir)
    if not golden_files:
        print(
            f"Error: No image files found in golden directory '{golden_dir}'. "
            f"Need at least 1 known-good image (20-30 recommended per AGENTS.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {len(golden_files)} golden image(s) from {golden_dir}...")
    golden_images = []
    for gf in golden_files:
        try:
            im = load_image(gf)
            golden_images.append(im)
        except Exception as e:
            print(f"Warning: Failed to load golden image {gf}: {e}", file=sys.stderr)

    if not golden_images:
        print("Error: Could not load any valid golden images.", file=sys.stderr)
        sys.exit(1)

    # Build part mask from primary golden image
    part_mask = build_part_mask(golden_images[0])

    # Build statistical golden reference
    golden_ref = GoldenReference(mm_per_px=mm_per_px)
    golden_ref.build(golden_images, part_mask=part_mask)

    # Instantiate Calibrated Engine
    engine = CalibratedEngine(golden_ref, sensitivity=sensitivity, roi_map=roi_map,
                               min_registration_confidence=min_reg_conf)

    # Calibrate noise floor if 2+ golden images exist
    if len(golden_images) >= 2:
        print(
            f"Calibrating noise floor with {len(golden_images)} golden images (percentile={noise_percentile})..."
        )
        engine.calibrate_noise_floor(golden_images, percentile=noise_percentile)
    else:
        print(
            f"Notice: Only 1 golden image provided ({golden_files[0].name}). "
            "Falling back to spatial variance estimation. "
            "Noise floor calibration skipped (requires 2+ golden images per AGENTS.md rule 3 & 4)."
        )

    print(f"\
Starting inspection on {len(input_files)} part(s)...\
")

    # Run Inspection per part
    for in_file in input_files:
        try:
            sample_img = load_image(in_file)
        except Exception as e:
            print(f"Error loading {in_file}: {e}", file=sys.stderr)
            continue

        res = engine.inspect_ranked(sample_img, top_n=args.top_n)

        # Print report
        print(format_ranked_report(res, name=in_file.name, top_n=args.top_n))

        # Save annotated image
        annotated_path = annotated_dir / f"{in_file.stem}_annotated.png"
        if res.annotated_image is not None:
            cv2.imwrite(str(annotated_path), res.annotated_image)
            print(f"  -> Saved annotated image to: {annotated_path}")

        # Save JSON result per part
        report_json = {
            "part_id": spec.get("part_id", "UNKNOWN"),
            "file": in_file.name,
            "verdict": res.verdict,
            "gate_failures": res.gate_failures,
            "global_metrics": res.global_metrics,
            "detector_stats": res.detector_stats,
            "total_candidates": len(res.defects),
            "top_n_shown": min(args.top_n, len(res.defects)),
            "defects": [d.to_dict() for d in res.defects],
        }
        json_path = reports_dir / f"{in_file.stem}_report.json"
        with open(json_path, "w") as f:
            json.dump(report_json, f, indent=2)
        print(f"  -> Saved JSON report to: {json_path}\
")


def main():
    args = parse_args()

    # 1. Load Part Specification
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Part specification not found at {spec_path}", file=sys.stderr)
        sys.exit(1)

    with open(spec_path, "r") as f:
        spec = json.load(f)

    # 2. Setup Output Directories
    out_dir = Path(args.out)
    annotated_dir = out_dir / "annotated"
    reports_dir = out_dir / "reports"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 3. Collect Input Images
    input_path = Path(args.input)
    input_files = find_image_files(input_path)
    if not input_files:
        print(
            f"Error: No input images found matching '{input_path}'.", file=sys.stderr
        )
        sys.exit(1)

    # 4. Dispatch based on --consensus
    if args.consensus:
        run_consensus_inspection(
            input_files=input_files,
            spec=spec,
            args=args,
            annotated_dir=annotated_dir,
            reports_dir=reports_dir,
        )
    else:
        run_standard_inspection(
            input_files=input_files,
            spec=spec,
            args=args,
            annotated_dir=annotated_dir,
            reports_dir=reports_dir,
        )


if __name__ == "__main__":
    main()
