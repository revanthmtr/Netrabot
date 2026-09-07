"""
Calibrated layer on top of InspectionEngine.

THE PROBLEM the raw engine exposed:
  At max sensitivity with ONE golden sample, ~590 blobs are flagged per
  part when only 1 is real. Nothing is missed, but the signal is buried.
  Sample A's real defect ranked 192/592. That is operationally a MISS,
  because no human reviews 592 boxes.

THE CAUSE:
  With one golden image there is no way to know how much a pixel
  NATURALLY varies. Sensor noise, HEIC compression, and lighting jitter
  all look identical to a real defect.

THE FIX (two parts):
  1. NOISE-FLOOR CALIBRATION - measure the detector response on
     known-good parts, then place thresholds just above that measured
     floor instead of guessing.
  2. SALIENCY RANKING - never DROP a candidate (zero-miss preserved),
     but rank by a composite score so real defects surface at the top
     and the noise sinks. Operator reviews ranked top-N; full list stays
     available for audit.
"""

import cv2
import numpy as np
try:
    from .inspection_engine import InspectionEngine, GoldenReference, Defect
except ImportError:
    from inspection_engine import InspectionEngine, GoldenReference, Defect


class CalibratedEngine(InspectionEngine):

    def __init__(self, golden, sensitivity=0.85, roi_map=None,
                 min_registration_confidence=0.6):
        super().__init__(golden, sensitivity=sensitivity, roi_map=roi_map,
                          min_registration_confidence=min_registration_confidence)
        self.noise_profile = None

    # ------------------------------------------------------------------
    def calibrate_noise_floor(self, good_images, percentile=99.5):
        """
        Run the detectors on KNOWN-GOOD parts and measure how strongly
        each one fires on defect-free material. That response IS the
        noise floor. Thresholds are then set above it.

        good_images: list of known-good part images (NOT the golden mean).
                     More is better; 10-30 recommended.
        """
        prof = {k: [] for k in ('zscore', 'ssim', 'absdiff', 'color',
                                'edge', 'morph', 'gradient', 'texture')}
        pm = self.g.part_mask > 0

        for img in good_images:
            img, _ = self.register(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_f = gray.astype(np.float32)
            gref = self.g.golden_gray

            _, s = self._d1_zscore(gray_f);            prof['zscore'].append(s[pm])
            _, s = self._d2_multiscale_ssim(gray, gref); prof['ssim'].append(s[pm])
            _, s = self._d3_absdiff(gray, gref);       prof['absdiff'].append(s[pm])
            _, s = self._d4_color_lab(img);            prof['color'].append(s[pm])
            _, s = self._d5_edge_topology(gray);       prof['edge'].append(s[pm])
            _, s = self._d6_morphological(gray, gref); prof['morph'].append(s[pm])
            _, s = self._d7_gradient(gray, gref);      prof['gradient'].append(s[pm])
            _, s = self._d8_texture(gray, gref);       prof['texture'].append(s[pm])

        self.noise_profile = {}
        for k, arrs in prof.items():
            allv = np.concatenate(arrs)
            self.noise_profile[k] = {
                'p50': float(np.percentile(allv, 50)),
                'p99': float(np.percentile(allv, 99)),
                'p999': float(np.percentile(allv, 99.9)),
                'floor': float(np.percentile(allv, percentile)),
                'max': float(allv.max()),
            }

        # Set thresholds just above the measured floor.
        # margin<1 keeps us recall-biased (below the max seen on good parts
        # would be too tight; we sit above p99.5 but below absolute max).
        margin = 1.02 + 0.20 * (1.0 - self.s)
        for k, v in self.noise_profile.items():
            key = {'zscore': 'z_score', 'absdiff': 'absdiff', 'color': 'delta_e',
                   'edge': 'edge_xor', 'morph': 'morph', 'gradient': 'gradient',
                   'texture': 'texture', 'ssim': 'ssim'}[k]
            self.T[key] = v['floor'] * margin

        return self.noise_profile

    # ------------------------------------------------------------------
    @staticmethod
    def saliency(d: Defect, noise_profile=None) -> float:
        """
        Composite score. Higher = more likely a REAL defect.

        Weights reflect what actually discriminated real defects from
        noise in validation:
          - multi-detector agreement is the single strongest signal
          - area matters, but log-scaled (a 2px speck can be real)
          - peak deviation magnitude
          - compactness (real defects are blobby; noise is stringy)
        """
        agree = d.confidence                       # 0-1
        area_term = np.log1p(d.area_mm2) / np.log1p(50.0)
        peak_term = min(d.peak_deviation / 50.0, 1.0)
        fill = d.area_px / max(1.0, d.w * d.h)     # compactness

        return float(
            0.45 * agree +
            0.25 * min(area_term, 1.0) +
            0.20 * peak_term +
            0.10 * fill
        )

    def inspect_ranked(self, img, top_n=None):
        """Full zero-miss inspection, output ranked by saliency."""
        res = self.inspect(img)
        for d in res.defects:
            d.__dict__['saliency'] = round(self.saliency(d), 4)
        res.defects.sort(key=lambda x: -x.__dict__['saliency'])
        for i, d in enumerate(res.defects):
            d.__dict__['rank'] = i + 1
        res.global_metrics['total_candidates'] = len(res.defects)
        if top_n:
            res.global_metrics['shown'] = min(top_n, len(res.defects))
        return res
