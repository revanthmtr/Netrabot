"""
Zero-Miss Visual Inspection Engine
==================================
Design principle: MAXIMUM RECALL. A missed defect is unacceptable;
a false alarm is merely expensive. Every detector runs independently
and results are fused with OR-logic, so a defect only needs to be
caught by ONE detector to be reported.

Detectors:
  D1  Statistical Z-score      - deviation vs per-pixel golden mean/std
  D2  Multi-scale SSIM         - structural anomaly at 4 window sizes
  D3  Absolute difference      - raw intensity change
  D4  Lab color / dEab         - perceptual color deviation per pixel
  D5  Edge topology            - Canny XOR, catches shape/line breaks
  D6  Morphological residue    - blackhat/tophat, catches specks & bridges
  D7  Gradient orientation     - Sobel angle change, catches subtle warps
  D8  Local contrast / texture - std-dev map, catches fade & smear
  D9  Connectivity topology    - contour count/euler, catches ink bridges

Fusion: pixel flagged if ANY detector fires (union), then blobs are
scored by how many detectors agree (confidence), NOT filtered by it.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Defect:
    """A single detected defect."""
    id: int
    x: int
    y: int
    w: int
    h: int
    area_px: float
    area_mm2: float
    centroid: tuple
    detectors: List[str]          # which detectors fired here
    confidence: float             # 0-1, fraction of detectors agreeing
    severity: str                 # CRITICAL / MAJOR / MINOR
    defect_type: str              # classified type
    peak_deviation: float         # strongest signal at this location
    region: str                   # which ROI it falls in
    measurements: Dict[str, Dict] = field(default_factory=dict)
    # per-detector quantitative evidence: {detector: {label, value,
    # threshold, exceeds_by_pct}} -- the actual measured parameter that
    # crossed the line, not just the detector's name.

    def to_dict(self):
        d = self.__dict__.copy()
        d['centroid'] = list(self.centroid)
        return d


@dataclass
class InspectionResult:
    verdict: str                          # PASS / FAIL / REVIEW
    defects: List[Defect] = field(default_factory=list)
    detector_stats: Dict = field(default_factory=dict)
    gate_failures: List[str] = field(default_factory=list)
    global_metrics: Dict = field(default_factory=dict)
    annotated_image: Optional[np.ndarray] = None


# ----------------------------------------------------------------------
# Golden reference (statistical, built from N good samples)
# ----------------------------------------------------------------------

class GoldenReference:
    """
    Statistical model of a good part.

    CRITICAL: built from MULTIPLE good samples, not one master.
    Per-pixel std tells us how much natural variation is normal at
    each location -- edges vary a lot, flat areas vary little. This
    is what lets us use a tight threshold without drowning in false
    positives at high-texture areas.
    """

    def __init__(self, mm_per_px: float = 1.0):
        self.mm_per_px = mm_per_px
        self.mean_gray = None
        self.std_gray = None
        self.mean_bgr = None
        self.std_bgr = None
        self.mean_lab = None
        self.edge_prob = None       # probability an edge exists at each px
        self.mean_contour_count = None
        self.std_contour_count = None
        self.n_samples = 0
        self.part_mask = None       # inspect only inside the part

    def build(self, good_images: List[np.ndarray], part_mask: np.ndarray = None):
        """Build statistical model from good samples."""
        if len(good_images) < 1:
            raise ValueError("Need at least 1 good image")

        # Standardize all images to the dimensions of the primary reference image
        target_h, target_w = good_images[0].shape[:2]
        standardized = []
        for im in good_images:
            if im.shape[:2] != (target_h, target_w):
                standardized.append(cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA))
            else:
                standardized.append(im)
        good_images = standardized

        n = len(good_images)
        self.n_samples = n

        grays = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
                          for im in good_images])
        bgrs = np.stack([im.astype(np.float32) for im in good_images])
        labs = np.stack([rgb2lab(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0)
                         for im in good_images])

        self.mean_gray = grays.mean(axis=0)
        self.mean_bgr = bgrs.mean(axis=0)
        self.mean_lab = labs.mean(axis=0)

        if n >= 2:
            self.std_gray = grays.std(axis=0)
            self.std_bgr = bgrs.std(axis=0)
        else:
            # Single-sample fallback: estimate local variance spatially
            # instead of across samples. Less accurate but functional.
            g = grays[0]
            local_mean = cv2.blur(g, (9, 9))
            local_sq = cv2.blur(g * g, (9, 9))
            self.std_gray = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
            self.std_bgr = np.stack([self.std_gray]*3, axis=-1)

        # Edge probability map
        edges = np.stack([cv2.Canny(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 50, 150)
                          for im in good_images]).astype(np.float32) / 255.0
        self.edge_prob = edges.mean(axis=0)

        # Contour count distribution
        counts = []
        for im in good_images:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            counts.append(len([c for c in cnts if cv2.contourArea(c) > 20]))
        self.mean_contour_count = float(np.mean(counts))
        self.std_contour_count = float(np.std(counts)) if n >= 2 else 2.0

        self.part_mask = part_mask if part_mask is not None else \
            np.ones(self.mean_gray.shape, dtype=np.uint8) * 255

        return self

    def build_from_median(self, images: List[np.ndarray], part_mask: np.ndarray = None):
        """
        Build statistical golden reference using the per-pixel MEDIAN of N samples.
        Used in consensus mode where multiple samples of the same part type are available,
        so isolated defects on individual parts are rejected by the median.
        """
        if len(images) < 1:
            raise ValueError("Need at least 1 image to build median reference")

        # Standardize all images to the dimensions of the primary reference image
        target_h, target_w = images[0].shape[:2]
        standardized = []
        for im in images:
            if im.shape[:2] != (target_h, target_w):
                standardized.append(cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA))
            else:
                standardized.append(im)
        images = standardized

        n = len(images)
        self.n_samples = n

        grays = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
                          for im in images])
        bgrs = np.stack([im.astype(np.float32) for im in images])
        labs = np.stack([rgb2lab(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0)
                         for im in images])

        self.mean_gray = np.median(grays, axis=0)
        self.mean_bgr = np.median(bgrs, axis=0)
        self.mean_lab = np.median(labs, axis=0)

        # Estimate variance combining across-sample variance and spatial variance
        g = self.mean_gray
        local_mean = cv2.blur(g, (9, 9))
        local_sq = cv2.blur(g * g, (9, 9))
        spatial_std = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
        sample_std = grays.std(axis=0) if n >= 2 else spatial_std

        self.std_gray = np.maximum(sample_std, spatial_std)
        self.std_bgr = np.stack([self.std_gray] * 3, axis=-1)

        # Edge probability map
        edges = np.stack([cv2.Canny(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 50, 150)
                          for im in images]).astype(np.float32) / 255.0
        self.edge_prob = edges.mean(axis=0)

        # Contour count distribution
        counts = []
        for im in images:
            g_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            counts.append(len([c for c in cnts if cv2.contourArea(c) > 20]))
        self.mean_contour_count = float(np.median(counts))
        self.std_contour_count = float(np.std(counts)) if n >= 2 else 2.0

        self.part_mask = part_mask if part_mask is not None else \
            np.ones(self.mean_gray.shape, dtype=np.uint8) * 255

        return self

    @property
    def golden_bgr(self):
        return self.mean_bgr.astype(np.uint8)

    @property
    def golden_gray(self):
        return self.mean_gray.astype(np.uint8)


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------

class InspectionEngine:
    """
    Multi-detector fusion engine tuned for maximum recall.

    Sensitivity is controlled by `sensitivity` (0-1):
      1.0 = maximum recall, most false positives
      0.5 = balanced
      0.0 = conservative
    Default 0.85 = strongly recall-biased.
    """

    def __init__(self, golden: GoldenReference, sensitivity: float = 0.85,
                 min_defect_mm2: float = 0.0, roi_map: Dict = None,
                 min_registration_confidence: float = 0.6):
        self.g = golden
        self.s = np.clip(sensitivity, 0.0, 1.0)
        self.min_defect_mm2 = min_defect_mm2
        self.roi_map = roi_map or {}
        # Phase-correlation response below this means the sample could not
        # be reliably aligned to the golden reference (severe exposure/
        # lighting mismatch, camera shake, wrong part). Observed directly
        # on a real sample: response 0.43 (vs 0.98-0.99 for a normal
        # capture) produced 5366 false candidates from edge jitter alone,
        # burying the one real defect. Running the 9 detectors on a
        # misaligned frame is meaningless, so that case is refused rather
        # than scored -- see inspect().
        self.min_registration_confidence = min_registration_confidence

        # Thresholds scale inversely with sensitivity.
        # At s=1.0 these are aggressive; at s=0.0 conservative.
        self.T = {
            'z_score':      6.0 - 3.5 * self.s,    # 2.5 at max sens
            'ssim':         0.35 - 0.20 * self.s,   # local SSIM drop
            'absdiff':      45 - 30 * self.s,       # 15 at max sens
            'delta_e':      12.0 - 8.5 * self.s,    # 3.5 at max sens
            'edge_xor':     0.55 - 0.30 * self.s,
            'morph':        40 - 25 * self.s,
            'gradient':     55 - 30 * self.s,
            'texture':      28 - 18 * self.s,
            'min_blob_px':  max(2, int(20 - 18 * self.s)),  # 2px at max sens
        }

    # ---------------- Stage 1: quality gate ----------------

    def quality_gate(self, img: np.ndarray) -> List[str]:
        """
        Reject unusable images rather than silently passing them.

        Sharpness/exposure/clipping are measured over the PRODUCT region
        only (part_mask), never the studio background. A white backdrop
        around a black part always reads as "highlight clipped" and a
        dark backdrop around a light part always reads as "shadow
        clipped" regardless of the product's actual condition -- observed
        directly: every real sample photographed against this rig's white
        background showed 10-14% highlight clipping purely from the
        background, even on genuinely good parts.
        """
        fails = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pm = self.g.part_mask
        masked = pm is not None and pm.shape == gray.shape
        region = gray[pm > 0] if masked else gray.ravel()

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_golden = cv2.Laplacian(self.g.golden_gray, cv2.CV_64F)
        if masked:
            sharp = float(lap[pm > 0].var())
            golden_sharp = float(lap_golden[pm > 0].var())
        else:
            sharp = float(lap.var())
            golden_sharp = float(lap_golden.var())
        if sharp < golden_sharp * 0.55:
            fails.append(f"BLUR: sharpness {sharp:.0f} vs golden {golden_sharp:.0f}")

        mean_v = region.mean()
        if mean_v < 35:
            fails.append(f"UNDEREXPOSED: mean {mean_v:.0f}")
        if mean_v > 225:
            fails.append(f"OVEREXPOSED: mean {mean_v:.0f}")

        clipped_hi = (region >= 254).sum() / region.size
        clipped_lo = (region <= 1).sum() / region.size
        if clipped_hi > 0.10:
            fails.append(f"HIGHLIGHT CLIPPING: {clipped_hi*100:.1f}%")
        if clipped_lo > 0.15:
            fails.append(f"SHADOW CLIPPING: {clipped_lo*100:.1f}%")

        if img.shape[:2] != self.g.mean_gray.shape:
            # Check if sub-crop that aligns to golden reference or auto-standardize
            g_h, g_w = self.g.mean_gray.shape
            h, w = img.shape[:2]
            is_subcrop = False
            if h <= g_h and w <= g_w:
                gray_u8 = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
                res_m = cv2.matchTemplate(self.g.golden_gray, gray_u8, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, _ = cv2.minMaxLoc(res_m)
                if max_v >= 0.70:
                    is_subcrop = True
            if not is_subcrop:
                img_std = cv2.resize(img, (g_w, g_h), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
                region = gray[pm > 0] if (pm is not None and pm.shape == gray.shape) else gray.ravel()

        return fails

    # ---------------- Stage 2: registration ----------------

    def register(self, img: np.ndarray, max_shift_px: float = 40.0):
        """
        Sub-pixel translation correction ONLY (fixed-rig assumption).

        Deliberately does NOT use homography: a full perspective warp
        would silently absorb genuine position/rotation defects. Shift
        beyond tolerance is REPORTED as a defect, not corrected away.
        """
        if img.shape[:2] != self.g.mean_gray.shape:
            g_h, g_w = self.g.mean_gray.shape
            h, w = img.shape[:2]
            is_subcrop = False
            if h <= g_h and w <= g_w:
                gray_sub = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res_match = cv2.matchTemplate(self.g.golden_gray, gray_sub, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res_match)
                if max_val >= 0.70:
                    mx, my = max_loc
                    canvas = np.copy(self.g.golden_bgr)
                    canvas[my:my+h, mx:mx+w] = img
                    img = canvas
                    is_subcrop = True
            if not is_subcrop:
                img = cv2.resize(img, (g_w, g_h), interpolation=cv2.INTER_AREA)

        if img.shape[:2] != self.g.mean_gray.shape:
            info = {
                'dx': 0.0, 'dy': 0.0,
                'shift_px': 999.0,
                'shift_mm': 999.0 * self.g.mm_per_px,
                'response': 0.0,
                'exceeded': True,
                'error': f"SIZE MISMATCH: {img.shape[:2]} vs {self.g.mean_gray.shape}"
            }
            return img, info

        gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32)
        gref = self.g.mean_gray.astype(np.float32)

        # Phase correlation's response is a genuine confidence measure
        # (5x5 peak-centroid power, 1.0 = single sharp peak) but it is
        # phase-only and does discard amplitude, so real exposure/lighting
        # differences between golden and sample legitimately suppress it
        # even at zero true shift -- confirmed against a real sample:
        # response 0.43 at a measured 0.12px shift, purely from 14.5%
        # highlight clipping. CLAHE-normalize ONLY the pair fed to
        # phaseCorrelate so illumination stops masking a real alignment;
        # the returned (dx, dy) still gets applied to the untouched
        # original `img` below, so detectors never see equalized pixels.
        # Validated: a genuinely uncorrelated pair's response still drops
        # under CLAHE (does not falsely rescue noise), and a genuine 15px
        # shift is reported identically with or without it (does not mask
        # a real position defect).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray_u8).astype(np.float32)
        gref_eq = clahe.apply(self.g.golden_gray).astype(np.float32)

        win = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(gref_eq * win, gray_eq * win)

        shift_mag = float(np.hypot(dx, dy))
        info = {'dx': float(dx), 'dy': float(dy),
                'shift_px': shift_mag,
                'shift_mm': shift_mag * self.g.mm_per_px,
                'response': float(response),
                'exceeded': shift_mag > max_shift_px}

        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return aligned, info

    # ---------------- Stage 3: the 9 detectors ----------------

    def _d1_zscore(self, gray_f):
        """Statistical deviation from golden mean, normalized by local std."""
        std = np.maximum(self.g.std_gray, 2.0)   # floor prevents div blowup
        z = np.abs(gray_f - self.g.mean_gray) / std
        return (z > self.T['z_score']).astype(np.uint8) * 255, z

    def _d2_multiscale_ssim(self, gray, gref):
        """SSIM at several window sizes: small wins catch specks,
        large wins catch smears/fades."""
        combined = np.zeros(gray.shape, dtype=np.uint8)
        peak = np.zeros(gray.shape, dtype=np.float32)
        I1 = gray.astype(np.float32)
        I2 = gref.astype(np.float32)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        for win in (7, 15, 31):
            ksize = (win, win)
            sigma = 1.5
            mu1 = cv2.GaussianBlur(I1, ksize, sigma)
            mu2 = cv2.GaussianBlur(I2, ksize, sigma)
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.GaussianBlur(I1 * I1, ksize, sigma) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(I2 * I2, ksize, sigma) - mu2_sq
            sigma12 = cv2.GaussianBlur(I1 * I2, ksize, sigma) - mu1_mu2
            smap = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-7)

            dev = (1.0 - smap).astype(np.float32)
            peak = np.maximum(peak, dev)
            combined |= (dev > self.T['ssim']).astype(np.uint8) * 255
        return combined, peak

    def _d3_absdiff(self, gray, gref):
        d = cv2.absdiff(gray, gref)
        return (d > self.T['absdiff']).astype(np.uint8) * 255, d.astype(np.float32)

    def _d4_color_lab(self, img):
        """Perceptual color deviation (CIEDE2000-adjacent, dEab for speed)."""
        lab = rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0)
        de = np.sqrt(((lab - self.g.mean_lab) ** 2).sum(axis=2))
        return (de > self.T['delta_e']).astype(np.uint8) * 255, de.astype(np.float32)

    def _d5_edge_topology(self, gray):
        """Edges present in sample but not golden (or vice versa)."""
        e = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        # deviation from expected edge probability
        dev = np.abs(e - self.g.edge_prob)
        return (dev > self.T['edge_xor']).astype(np.uint8) * 255, dev

    def _d6_morphological(self, gray, gref):
        """Blackhat/tophat residue: excellent for small dark specks
        (sample B) and bright voids/pinholes."""
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bh_s = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        bh_g = cv2.morphologyEx(gref, cv2.MORPH_BLACKHAT, k)
        th_s = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        th_g = cv2.morphologyEx(gref, cv2.MORPH_TOPHAT, k)
        resid = cv2.max(cv2.absdiff(bh_s, bh_g), cv2.absdiff(th_s, th_g))
        return (resid > self.T['morph']).astype(np.uint8) * 255, resid.astype(np.float32)

    def _d7_gradient(self, gray, gref):
        """Gradient magnitude change - catches subtle line-weight and
        warp differences invisible to intensity diff."""
        def gmag(im):
            gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)
        d = np.abs(gmag(gray) - gmag(gref))
        return (d > self.T['gradient']).astype(np.uint8) * 255, d

    def _d8_texture(self, gray, gref):
        """Local standard deviation map difference - catches fading,
        smearing, and print-density loss over an area."""
        def local_std(im):
            f = im.astype(np.float32)
            m = cv2.blur(f, (11, 11))
            sq = cv2.blur(f * f, (11, 11))
            return np.sqrt(np.maximum(sq - m * m, 0))
        d = np.abs(local_std(gray) - local_std(gref))
        return (d > self.T['texture']).astype(np.uint8) * 255, d

    def _d9_topology(self, gray):
        """Global connectivity check - catches ink bridges that merge
        two shapes (sample C) even when the pixel change is tiny."""
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        count = len([c for c in cnts if cv2.contourArea(c) > 20])
        tol = max(1.0, 3.0 * self.g.std_contour_count)
        deviated = abs(count - self.g.mean_contour_count) > tol
        return count, deviated, tol

    # ---------------- quantitative evidence per detector ----------------

    _METRIC_META = {
        'zscore':   ('z_score',  'statistical deviation (Z-score)', 'sigma'),
        'ssim':     ('ssim',     'structural dissimilarity (1-SSIM)', ''),
        'absdiff':  ('absdiff',  'raw intensity change', 'gray levels (0-255)'),
        'color':    ('delta_e',  'perceptual color deviation (CIE Lab dE)', 'dE'),
        'edge':     ('edge_xor', 'edge probability deviation', ''),
        'morph':    ('morph',    'morphological residue (speck/void)', ''),
        'gradient': ('gradient', 'gradient magnitude change', ''),
        'texture':  ('texture',  'local texture / density deviation', ''),
    }

    @staticmethod
    def _measurements(fired, sigs, comp, T):
        """
        For every detector that fired on this blob, report the actual
        measured value in that region and the threshold it crossed --
        the concrete parameter that makes this a defect, not just a
        detector name. This is what an operator (or a reviewer who
        cannot see the defect by eye) uses to judge the call.
        """
        out = {}
        for k in fired:
            if k not in sigs or k not in InspectionEngine._METRIC_META:
                continue
            tkey, label, unit = InspectionEngine._METRIC_META[k]
            if tkey not in T:
                continue
            val = float(sigs[k][comp].max())
            thr = float(T[tkey])
            out[k] = {
                'label': label,
                'unit': unit,
                'value': round(val, 3),
                'threshold': round(thr, 3),
                'exceeds_by_pct': round(100.0 * (val / thr - 1.0), 1) if thr else None,
            }
        return out

    @staticmethod
    def top_measurement(measurements: Dict) -> Optional[str]:
        """Single most-exceeding metric, formatted for a one-line summary."""
        if not measurements:
            return None

        def _get_sort_key(kv):
            m = kv[1]
            if not isinstance(m, dict):
                return 0.0
            if m.get('exceeds_by_pct') is not None:
                try:
                    return float(m['exceeds_by_pct'])
                except (ValueError, TypeError):
                    pass
            if m.get('delta') is not None:
                try:
                    return float(m['delta'])
                except (ValueError, TypeError):
                    pass
            return 0.0

        k, m = max(measurements.items(), key=_get_sort_key)
        if not isinstance(m, dict):
            return str(m)

        unit = f" {m['unit']}" if m.get('unit') else ""
        val = m.get('value', 'N/A')
        thr = m.get('threshold', 'N/A')
        label = m.get('label', k)

        if m.get('exceeds_by_pct') is not None:
            pct_info = f", +{m['exceeds_by_pct']}%"
        elif m.get('delta') is not None:
            pct_info = f", delta: +{m['delta']}"
        else:
            pct_info = ""

        return f"{label}: {val}{unit} (tolerance {thr}{unit}{pct_info})"

    # ---------------- Stage 4: fusion + blob extraction ----------------

    def inspect(self, img: np.ndarray, do_register: bool = True) -> InspectionResult:
        res = InspectionResult(verdict="PASS")

        # Gate
        res.gate_failures = self.quality_gate(img)
        if res.gate_failures:
            res.verdict = "REVIEW"

        # Register
        reg_info = {}
        if do_register:
            img, reg_info = self.register(img)
            res.global_metrics['registration'] = reg_info
            if reg_info.get('exceeded'):
                res.verdict = "FAIL"
                res.gate_failures.append(
                    f"POSITION OUT OF TOLERANCE: {reg_info['shift_px']:.1f}px")
            elif reg_info.get('response', 1.0) < self.min_registration_confidence:
                # Alignment itself could not be trusted -- do not run the
                # detectors on a misaligned frame, they will fire on edge
                # jitter everywhere and bury (or misreport) the real defect.
                res.verdict = "REVIEW"
                res.gate_failures.append(
                    f"REGISTRATION CONFIDENCE TOO LOW: {reg_info['response']:.3f} "
                    f"(need >= {self.min_registration_confidence}) -- alignment "
                    f"unreliable, image not scored for defects")
                return res

        if img.shape[:2] != self.g.mean_gray.shape:
            res.verdict = "FAIL"
            h, w = img.shape[:2]
            g_h, g_w = self.g.mean_gray.shape
            res.defects.append(Defect(
                id=1,
                x=0, y=0, w=w, h=h,
                area_px=float(w * h),
                area_mm2=float(w * h) * (self.g.mm_per_px ** 2),
                centroid=(float(w / 2.0), float(h / 2.0)),
                detectors=["quality_gate"],
                confidence=1.0,
                severity="CRITICAL",
                defect_type="DIMENSION_MISMATCH",
                peak_deviation=255.0,
                region="unassigned",
                measurements={'dimension': {
                    'label': 'image dimensions vs golden',
                    'unit': 'px',
                    'value': f"{w}x{h}",
                    'threshold': f"{g_w}x{g_h}",
                }},
            ))
            return res

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        gref = self.g.golden_gray

        # Run all detectors
        masks, sigs = {}, {}
        masks['zscore'],   sigs['zscore']   = self._d1_zscore(gray_f)
        masks['ssim'],     sigs['ssim']     = self._d2_multiscale_ssim(gray, gref)
        masks['absdiff'],  sigs['absdiff']  = self._d3_absdiff(gray, gref)
        masks['color'],    sigs['color']    = self._d4_color_lab(img)
        masks['edge'],     sigs['edge']     = self._d5_edge_topology(gray)
        masks['morph'],    sigs['morph']    = self._d6_morphological(gray, gref)
        masks['gradient'], sigs['gradient'] = self._d7_gradient(gray, gref)
        masks['texture'],  sigs['texture']  = self._d8_texture(gray, gref)

        contour_count, topo_dev, topo_tol = self._d9_topology(gray)
        res.global_metrics['topology'] = {
            'contour_count': contour_count,
            'golden_mean': self.g.mean_contour_count,
            'tolerance': topo_tol,
            'deviated': bool(topo_dev),
        }

        # Restrict to part area
        pm = self.g.part_mask
        for k in masks:
            masks[k] = cv2.bitwise_and(masks[k], masks[k], mask=pm)

        # UNION fusion - a defect need only be caught by ONE detector
        union = np.zeros(gray.shape, dtype=np.uint8)
        for m in masks.values():
            union = cv2.bitwise_or(union, m)

        # Gentle cleanup ONLY - do not erode away small real defects
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8))

        # Agreement map (how many detectors fired per pixel) -> confidence
        agree = np.zeros(gray.shape, dtype=np.float32)
        for m in masks.values():
            agree += (m > 0).astype(np.float32)
        n_det = len(masks)

        res.detector_stats = {
            k: {'flagged_px': int((m > 0).sum()),
                'pct': round(100.0 * (m > 0).sum() / max(1, pm.sum() / 255), 4)}
            for k, m in masks.items()
        }

        # Extract blobs
        n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(union, 8)
        defects = []
        did = 0
        for i in range(1, n_lbl):
            x, y, w, h, area = stats[i]
            if area < self.T['min_blob_px']:
                continue
            area_mm2 = area * (self.g.mm_per_px ** 2)
            if area_mm2 < self.min_defect_mm2:
                continue

            comp = (lbl == i)
            fired = [k for k, m in masks.items() if (m[comp] > 0).any()]
            conf = len(fired) / n_det

            peak = 0.0
            for k in ('zscore', 'ssim', 'color', 'absdiff'):
                if k in sigs:
                    peak = max(peak, float(sigs[k][comp].max()))

            did += 1
            defects.append(Defect(
                id=did, x=int(x), y=int(y), w=int(w), h=int(h),
                area_px=float(area), area_mm2=float(area_mm2),
                centroid=(float(cents[i][0]), float(cents[i][1])),
                detectors=fired, confidence=round(conf, 3),
                severity=self._severity(area_mm2, conf, fired),
                defect_type=self._classify(fired, sigs, comp, img),
                peak_deviation=round(peak, 2),
                region=self._which_region(x, y, w, h),
                measurements=self._measurements(fired, sigs, comp, self.T),
            ))

        # Topology defect (ink bridge) may have no strong pixel blob
        if topo_dev:
            defects.append(Defect(
                id=did + 1, x=0, y=0, w=0, h=0, area_px=0, area_mm2=0,
                centroid=(0, 0), detectors=['topology'], confidence=1.0,
                severity="MAJOR",
                defect_type="TOPOLOGY_CHANGE (merged/split elements - ink bridge or break)",
                peak_deviation=abs(contour_count - self.g.mean_contour_count),
                region="global",
                measurements={'topology': {
                    'label': 'connected element count',
                    'unit': 'elements',
                    'value': contour_count,
                    'threshold': round(self.g.mean_contour_count + topo_tol, 2),
                    'golden_mean': round(self.g.mean_contour_count, 2),
                    'exceeds_by_pct': round(100.0 * (abs(contour_count - self.g.mean_contour_count) / topo_tol - 1.0), 1) if topo_tol else None,
                }},
            ))

        defects.sort(key=lambda d: (-d.area_mm2, -d.confidence))
        res.defects = defects

        if defects and res.verdict == "PASS":
            res.verdict = "FAIL"
        # NOTE: if verdict is already "REVIEW" (quality_gate failed --
        # blur, bad exposure, clipping), it stays REVIEW even if defects
        # were found. A confident "FAIL" here would misreport "verified
        # defect" when what actually happened is "the photo itself can't
        # be trusted, so neither can anything the detectors found on it."
        # Observed directly: a 32.5%-highlight-clipped sample produced a
        # single "defect" covering ~64% of the whole image.

        res.annotated_image = self._annotate(img, defects)
        return res

    # ---------------- classification helpers ----------------

    @staticmethod
    def _severity(area_mm2, conf, fired):
        if 'color' in fired and area_mm2 > 5:
            return "CRITICAL"
        if area_mm2 > 2.0 or conf > 0.6:
            return "CRITICAL"
        if area_mm2 > 0.3 or conf > 0.35:
            return "MAJOR"
        return "MINOR"

    @staticmethod
    def _classify(fired, sigs, comp, img):
        f = set(fired)
        if 'color' in f and 'texture' in f and 'absdiff' not in f:
            return "COLOR_FADE / DISCOLORATION"
        if 'color' in f and len(f) <= 2:
            return "COLOR_DEVIATION"
        if 'morph' in f and 'absdiff' in f:
            return "SPECK / CONTAMINATION"
        if 'edge' in f and 'gradient' in f:
            return "EDGE_DEFECT / LINE_BREAK_OR_BRIDGE"
        if 'texture' in f and 'ssim' in f:
            return "SMEAR / PRINT_DENSITY_LOSS"
        if 'zscore' in f or 'ssim' in f:
            return "SURFACE_ANOMALY / SCRATCH_OR_MARK"
        return "UNCLASSIFIED_ANOMALY"

    def _which_region(self, x, y, w, h):
        cx, cy = x + w / 2, y + h / 2
        for name, box in self.roi_map.items():
            x1, y1, x2, y2 = box
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return name
        return "unassigned"

    @staticmethod
    def _annotate(img, defects):
        vis = img.copy()
        colors = {"CRITICAL": (0, 0, 255), "MAJOR": (0, 140, 255),
                  "MINOR": (0, 220, 255)}
        for d in defects:
            if d.w == 0:
                continue
            c = colors.get(d.severity, (255, 255, 255))
            pad = 6
            cv2.rectangle(vis, (d.x - pad, d.y - pad),
                          (d.x + d.w + pad, d.y + d.h + pad), c, 2)
            lbl = f"#{d.id} {d.severity[:4]} {d.area_mm2:.2f}mm2"
            ly = max(14, d.y - pad - 5)
            cv2.putText(vis, lbl, (max(2, d.x - pad), ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, lbl, (max(2, d.x - pad), ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        return vis


def report(res: InspectionResult, name: str = "") -> str:
    L = []
    L.append("=" * 68)
    L.append(f"INSPECTION REPORT{(' - ' + name) if name else ''}")
    L.append("=" * 68)
    L.append(f"VERDICT: {res.verdict}")
    if res.gate_failures:
        L.append("\nGATE / GLOBAL FAILURES:")
        for g in res.gate_failures:
            L.append(f"  ! {g}")

    reg = res.global_metrics.get('registration')
    if reg:
        L.append(f"\nRegistration: shift {reg['shift_px']:.2f}px "
                 f"({reg['shift_mm']:.3f}mm)  conf={reg['response']:.3f}")
    topo = res.global_metrics.get('topology')
    if topo:
        L.append(f"Topology: {topo['contour_count']} elements "
                 f"(golden {topo['golden_mean']:.1f} +/- tol {topo['tolerance']:.1f})"
                 f"{'  <-- DEVIATED' if topo['deviated'] else ''}")

    L.append(f"\nDETECTOR ACTIVITY:")
    for k, v in res.detector_stats.items():
        L.append(f"  {k:10s} {v['flagged_px']:>8d} px  ({v['pct']:.4f}%)")

    L.append(f"\nDEFECTS FOUND: {len(res.defects)}")
    if res.defects:
        L.append(f"{'ID':>3} {'SEV':<9} {'AREA mm2':>9} {'CONF':>5} "
                 f"{'POS':<14} {'TYPE':<42} DETECTORS")
        L.append("-" * 68)
        for d in res.defects:
            pos = f"({d.x},{d.y})" if d.w else "global"
            L.append(f"{d.id:>3} {d.severity:<9} {d.area_mm2:>9.3f} "
                     f"{d.confidence:>5.2f} {pos:<14} {d.defect_type:<42} "
                     f"{','.join(d.detectors)}")
            tm = InspectionEngine.top_measurement(d.measurements)
            if tm:
                L.append(f"      -> MEASURED: {tm}")
    L.append("=" * 68)
    return "\n".join(L)
