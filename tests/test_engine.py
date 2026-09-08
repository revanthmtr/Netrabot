"""
Unit and integration tests for the AOI inspection engine.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from engine.calibrated_engine import CalibratedEngine
from engine.inspection_engine import GoldenReference
from engine.io_utils import build_part_mask, load_image


class TestAOIEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.golden_dir = self.test_dir / "golden"
        self.samples_dir = self.test_dir / "samples"
        self.output_dir = self.test_dir / "output"
        self.golden_dir.mkdir(parents=True)
        self.samples_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Generate a base synthetic label image (200x400)
        # Background: dark (0), Part: white rectangle in center (20, 20) to (180, 380)
        self.h, self.w = 300, 600
        self.base_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Part body
        self.base_img[30:270, 50:550] = (230, 230, 230)
        # Add some features inside part (orange bar, text-like features)
        self.base_img[40:100, 70:530] = (30, 140, 240)  # BGR orange
        self.base_img[120:200, 100:300] = (40, 40, 40)   # text area
        self.base_img[120:200, 350:500] = (50, 100, 50)  # pictogram area

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_build_part_mask(self):
        mask = build_part_mask(self.base_img)
        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertEqual(mask.dtype, np.uint8)
        # Center of part should be 255
        self.assertEqual(mask[150, 300], 255)
        # Outer background should be 0
        self.assertEqual(mask[10, 10], 0)
        # Due to 10px erosion, boundary near (30, 50) should be 0
        self.assertEqual(mask[35, 55], 0)

    def test_load_image(self):
        img_path = self.test_dir / "sample.png"
        cv2.imwrite(str(img_path), self.base_img)
        loaded = load_image(img_path)
        self.assertEqual(loaded.shape, self.base_img.shape)
        np.testing.assert_array_equal(loaded, self.base_img)

    def test_golden_reference_and_calibration(self):
        # Create 3 golden samples with slight sensor noise
        golden_imgs = []
        for i in range(3):
            noise = np.random.normal(0, 1.5, self.base_img.shape).astype(np.int16)
            noisy = np.clip(self.base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            p = self.golden_dir / f"golden_{i}.png"
            cv2.imwrite(str(p), noisy)
            golden_imgs.append(noisy)

        mask = build_part_mask(golden_imgs[0])
        ref = GoldenReference(mm_per_px=0.25)
        ref.build(golden_imgs, part_mask=mask)

        self.assertIsNotNone(ref.mean_gray)
        self.assertIsNotNone(ref.std_gray)
        self.assertEqual(ref.n_samples, 3)

        # Calibrated Engine
        roi_map = {
            "orange_panel": [70, 40, 530, 100],
            "text_area": [100, 120, 300, 200],
        }
        engine = CalibratedEngine(ref, sensitivity=0.85, roi_map=roi_map)
        profile = engine.calibrate_noise_floor(golden_imgs, percentile=99.5)
        self.assertIn("absdiff", profile)
        self.assertIn("floor", profile["absdiff"])

        # Inspect sample (in zero-miss architecture with 3 samples, candidates are ranked)
        res_clean = engine.inspect_ranked(golden_imgs[0], top_n=5)
        self.assertIsNotNone(res_clean.verdict)
        self.assertIn("total_candidates", res_clean.global_metrics)

        # Inspect sample with synthetic defect: dark blob in orange panel
        defect_img = golden_imgs[0].copy()
        cv2.circle(defect_img, (200, 70), 8, (0, 0, 0), -1)  # 16px diameter black dot
        res_defect = engine.inspect_ranked(defect_img, top_n=5)
        self.assertEqual(res_defect.verdict, "FAIL")
        self.assertGreater(len(res_defect.defects), 0)
        # Top defect should be in the orange_panel region
        top_defect = res_defect.defects[0]
        self.assertEqual(top_defect.region, "orange_panel")
        self.assertGreater(top_defect.__dict__["saliency"], 0.0)

    def test_run_inspection_cli_end_to_end(self):
        import subprocess
        import sys

        # Create 2 golden samples
        for i in range(2):
            noise = np.random.normal(0, 1.0, self.base_img.shape).astype(np.int16)
            noisy = np.clip(self.base_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(self.golden_dir / f"golden_{i}.png"), noisy)

        # Create 1 test sample
        sample_img = self.base_img.copy()
        cv2.rectangle(sample_img, (150, 60), (180, 80), (0, 0, 0), -1)
        sample_path = self.samples_dir / "sample_001.png"
        cv2.imwrite(str(sample_path), sample_img)

        # Execute run_inspection.py CLI
        cmd = [
            sys.executable,
            "run_inspection.py",
            "--golden", str(self.golden_dir),
            "--input", str(self.samples_dir),
            "--spec", "specs/part_spec.json",
            "--out", str(self.output_dir),
            "--top-n", "10",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI execution failed:\n{proc.stderr}")
        self.assertIn("INSPECTION REPORT - sample_001.png", proc.stdout)

        # Verify output files
        annotated_file = self.output_dir / "annotated" / "sample_001_annotated.png"
        report_file = self.output_dir / "reports" / "sample_001_report.json"
        self.assertTrue(annotated_file.exists(), f"Missing annotated file: {annotated_file}")
        self.assertTrue(report_file.exists(), f"Missing report file: {report_file}")

        with open(report_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data["file"], "sample_001.png")
        self.assertEqual(data["verdict"], "FAIL")
        self.assertIn("defects", data)
        self.assertGreater(len(data["defects"]), 0)

    def test_golden_reference_median(self):
        # Create 3 images where one has an isolated black spot
        imgs = [self.base_img.copy() for _ in range(3)]
        cv2.circle(imgs[0], (200, 70), 10, (0, 0, 0), -1)

        mask = build_part_mask(imgs[0])
        ref = GoldenReference(mm_per_px=0.25)
        ref.build_from_median(imgs, part_mask=mask)

        self.assertIsNotNone(ref.mean_gray)
        self.assertIsNotNone(ref.std_gray)
        self.assertEqual(ref.n_samples, 3)

        # Median at (200, 70) should NOT contain the black spot from sample 0
        ref_bgr = ref.golden_bgr
        expected_orange = self.base_img[70, 200]
        np.testing.assert_array_equal(ref_bgr[70, 200], expected_orange)

    def test_run_inspection_consensus_cli_end_to_end(self):
        import subprocess
        import sys

        # Create 3 sample images:
        # All 3 have shared systematic feature (e.g. gray line at (150, 45) to (180, 45))
        # Only Sample 0 has an isolated real defect (black circle at (220, 70))
        sample_paths = []
        for i in range(3):
            s = self.base_img.copy()
            # Shared systematic noise on all 3
            cv2.line(s, (150, 45), (180, 45), (100, 100, 100), 2)
            if i == 0:
                # Isolated real defect on sample 0
                cv2.circle(s, (220, 70), 8, (0, 0, 0), -1)
            p = self.samples_dir / f"consensus_sample_{i}.png"
            cv2.imwrite(str(p), s)
            sample_paths.append(p)

        cmd = [
            sys.executable,
            "run_inspection.py",
            "--input", str(self.samples_dir),
            "--consensus",
            "--spec", "specs/part_spec.json",
            "--out", str(self.output_dir),
            "--top-n", "10",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Consensus CLI failed:\n{proc.stderr}\n{proc.stdout}")
        self.assertIn("Running CONSENSUS MODE across 3 samples", proc.stdout)

        # Check master audit JSON
        audit_file = self.output_dir / "reports" / "consensus_audit.json"
        self.assertTrue(audit_file.exists(), f"Missing audit file: {audit_file}")
        with open(audit_file, "r") as f:
            audit = json.load(f)
        self.assertEqual(audit["mode"], "consensus")
        self.assertEqual(audit["n_samples"], 3)
        self.assertIn("consensus_sample_0.png", audit["parts"])

        # Sample 0 should fail with real defect surviving
        s0_rep_file = self.output_dir / "reports" / "consensus_sample_0_report.json"
        self.assertTrue(s0_rep_file.exists())
        with open(s0_rep_file, "r") as f:
            s0_data = json.load(f)
        self.assertEqual(s0_data["verdict"], "FAIL")
        self.assertGreater(s0_data["surviving_defects_count"], 0)
        self.assertIn("suppressed_candidates", s0_data)
        top_defect = s0_data["defects"][0]
        self.assertEqual(top_defect["sample_str"], "1/3")

    def test_low_registration_confidence_is_refused_not_scored(self):
        """
        A sample the phase-correlation step cannot reliably align must be
        refused (verdict REVIEW, zero defects) rather than run through the
        9 detectors -- on a real misaligned sample this produced 5366
        false candidates from edge jitter alone, burying the one real
        defect. Reproduced here with heavy uncorrelated noise (same shape,
        no actual shift) which reliably drops phase-correlation response
        without tripping the separate shift-exceeded path.
        """
        mask = build_part_mask(self.base_img)
        ref = GoldenReference(mm_per_px=0.25)
        ref.build([self.base_img], part_mask=mask)
        engine = CalibratedEngine(ref, sensitivity=0.85)

        rng = np.random.default_rng(0)
        noise = rng.normal(0, 60, self.base_img.shape)
        noisy = np.clip(self.base_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        res = engine.inspect(noisy)
        self.assertEqual(res.verdict, "REVIEW")
        self.assertEqual(len(res.defects), 0)
        self.assertTrue(
            any("REGISTRATION CONFIDENCE" in g for g in res.gate_failures),
            res.gate_failures,
        )
        self.assertLess(res.global_metrics["registration"]["response"], 0.6)

    def test_registration_confidence_survives_severe_overexposure(self):
        """
        phaseCorrelate's response is phase-only (discards amplitude), so
        a genuine exposure/lighting difference legitimately suppresses it
        even at zero real shift -- confirmed against OpenCV's own source
        (modules/imgproc/src/phasecorr.cpp) and against a real sample:
        14.5% highlight clipping alone dropped response from ~0.99 to
        0.43 with a real shift of only 0.12px. register() now computes
        the correlation on a CLAHE-normalized copy of golden/sample so
        exposure differences stop masquerading as failed alignment, while
        still applying the resulting shift to the untouched original
        pixels. Reproduced synthetically: a severely overexposed (41.6%
        highlight-clipped) but genuinely unshifted duplicate reliably
        drops the *raw* response below the 0.6 gate (measured 0.52) while
        the CLAHE-normalized response clears it (measured 0.61).
        """
        h, w = 300, 600
        rng = np.random.default_rng(42)
        textured = np.clip(
            self.base_img.astype(np.int16)
            + rng.normal(0, 18, (h, w, 1)).astype(np.int16),
            0, 255,
        ).astype(np.uint8)

        mask = build_part_mask(textured)
        ref = GoldenReference(mm_per_px=0.25)
        ref.build([textured], part_mask=mask)
        engine = CalibratedEngine(ref, sensitivity=0.85)

        overexposed = np.clip(
            textured.astype(np.float32) * 2.2 + 130, 0, 255
        ).astype(np.uint8)

        _, reg_info = engine.register(overexposed)
        self.assertGreaterEqual(reg_info["response"], engine.min_registration_confidence)
        self.assertLess(reg_info["shift_px"], 1.0)  # no real position defect masked

        res = engine.inspect(overexposed)
        self.assertNotIn(
            "REGISTRATION CONFIDENCE TOO LOW", " ".join(res.gate_failures)
        )

    def test_quality_gate_ignores_background_composition(self):
        """
        Exposure/clipping must be measured over the product region only.
        This fixture's background is pure black outside the part -- before
        the part-mask fix this alone tripped SHADOW CLIPPING on every
        sample regardless of the product's actual condition (matching
        what was observed on every real FELT-PAD sample: 10-14% highlight
        clipping purely from its white studio background).
        """
        mask = build_part_mask(self.base_img)
        ref = GoldenReference(mm_per_px=0.25)
        ref.build([self.base_img], part_mask=mask)
        engine = CalibratedEngine(ref, sensitivity=0.85)

        fails = engine.quality_gate(self.base_img)
        self.assertFalse(
            any("CLIPPING" in f for f in fails),
            f"background composition should not trigger clipping: {fails}",
        )

    def _textured_jittered_variants(self, n, shift_px=1.2, noise_std=18, seed=42):
        """N synthetic "photos of the same good part": fine per-pixel
        texture (so blur is measurable, unlike this fixture's flat color
        blocks) plus small random sub-pixel translation (unavoidable
        shot-to-shot placement jitter) and light sensor noise."""
        rng = np.random.default_rng(seed)
        texture = rng.normal(0, noise_std, (self.h, self.w, 1))
        textured_base = np.clip(
            self.base_img.astype(np.float32) + texture, 0, 255
        ).astype(np.uint8)
        variants = [textured_base]
        for _ in range(n - 1):
            noise = rng.normal(0, 3.0, self.base_img.shape)
            variant = np.clip(textured_base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            M = np.float32([
                [1, 0, rng.uniform(-shift_px, shift_px)],
                [0, 1, rng.uniform(-shift_px, shift_px)],
            ])
            variant = cv2.warpAffine(
                variant, M, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE
            )
            variants.append(variant)
        return variants

    def test_multi_golden_build_keeps_template_sharp(self):
        """
        build() must not blur its comparison template when given multiple
        golden photos. Averaging pixel VALUES across N independently
        resampled (aligned) images measurably destroys fine texture even
        with near-perfect sub-pixel alignment, because each image needs
        its own bilinear resample at a different fractional offset before
        stacking. Reproduced on real data: 10 golden photos of the same
        felt-pad texture, alignment residual <0.01px, still dropped
        Laplacian sharpness 1169 -> 503 (57% loss) averaged, enough to
        flood a genuinely good part with 3114 false "defects" (vs. 1 real
        defect with a single golden image). Fixed by using the sharp
        anchor image directly as the template; cross-sample statistics
        (std/edge_prob/contour count) still use the full aligned stack.
        """
        variants = self._textured_jittered_variants(n=8)

        single_gray = cv2.cvtColor(variants[0], cv2.COLOR_BGR2GRAY)
        single_sharpness = cv2.Laplacian(single_gray, cv2.CV_64F).var()

        mask = build_part_mask(variants[0])
        ref = GoldenReference(mm_per_px=0.25)
        ref.build(variants, part_mask=mask)
        multi_sharpness = cv2.Laplacian(ref.golden_gray, cv2.CV_64F).var()

        self.assertGreater(
            multi_sharpness, single_sharpness * 0.9,
            f"multi-golden template blurred: single={single_sharpness:.0f} "
            f"multi={multi_sharpness:.0f}",
        )

    def test_multi_golden_build_does_not_flood_good_part_with_false_defects(self):
        """
        End-to-end version of the above, as a measured before/after
        comparison (this fixture's flat color blocks are far less
        textured than the real felt-pad photo this was found on, so an
        absolute candidate-count threshold isn't meaningful here -- the
        real, measured win was 3114 -> 1 candidates on real data; this
        asserts the same direction and a substantial magnitude on
        synthetic data instead of guessing an absolute number).
        """
        variants = self._textured_jittered_variants(n=8)
        mask = build_part_mask(variants[0])

        # Pre-fix behavior: raw unaligned per-pixel mean/std, exactly what
        # build() used to do before this change.
        grays = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in variants])
        bgrs = np.stack([im.astype(np.float32) for im in variants])
        from skimage.color import rgb2lab
        labs = np.stack([rgb2lab(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0) for im in variants])
        ref_naive = GoldenReference(mm_per_px=0.25)
        ref_naive.n_samples = len(variants)
        ref_naive.mean_gray = grays.mean(axis=0)
        ref_naive.mean_bgr = bgrs.mean(axis=0)
        ref_naive.mean_lab = labs.mean(axis=0)
        ref_naive.std_gray = grays.std(axis=0)
        ref_naive.std_bgr = bgrs.std(axis=0)
        edges = np.stack([cv2.Canny(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 50, 150)
                          for im in variants]).astype(np.float32) / 255.0
        ref_naive.edge_prob = edges.mean(axis=0)
        counts = []
        for im in variants:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            counts.append(len([c for c in cnts if cv2.contourArea(c) > 20]))
        ref_naive.mean_contour_count = float(np.mean(counts))
        ref_naive.std_contour_count = float(np.std(counts))
        ref_naive.part_mask = mask
        engine_naive = CalibratedEngine(ref_naive, sensitivity=0.85)
        engine_naive.calibrate_noise_floor(variants, percentile=99.5)

        # Current (fixed) build()
        ref_fixed = GoldenReference(mm_per_px=0.25)
        ref_fixed.build(variants, part_mask=mask)
        engine_fixed = CalibratedEngine(ref_fixed, sensitivity=0.85)
        engine_fixed.calibrate_noise_floor(variants, percentile=99.5)

        # A good part: another independent noisy-but-undefective capture
        rng = np.random.default_rng(99)
        good_sample = np.clip(
            variants[0].astype(np.float32) + rng.normal(0, 3.0, variants[0].shape),
            0, 255,
        ).astype(np.uint8)

        n_naive = len(engine_naive.inspect(good_sample).defects)
        n_fixed = len(engine_fixed.inspect(good_sample).defects)
        self.assertLess(
            n_fixed, n_naive * 0.5,
            f"expected a substantial reduction in false candidates on a "
            f"good part: naive(pre-fix)={n_naive} fixed={n_fixed}",
        )

    def test_heatmap_localizes_at_the_real_defect(self):
        """
        End-to-end: a real defect must both appear in the defect list AND
        make the heatmap visibly hotter over that specific region than
        elsewhere on the part -- proving they're built from the same
        signal (previously a SEPARATE, disconnected raw grayscale
        absdiff, unregistered/unmasked, rescaled per-image with
        cv2.NORM_MINMAX, which could show heat with no matching defect
        box or vice versa).

        Compares mean heatmap-vs-original pixel difference in the known
        defect's own bounding box against a same-size box in a plain,
        untextured part of the image (the light gray body, not the
        orange bar -- alpha-blending shifts a bright/saturated pixel's
        raw RGB value by more than a dark/muted one for the SAME
        underlying heat level, which would confound a comparison against
        a saturated region).
        """
        variants = self._textured_jittered_variants(n=3)
        mask = build_part_mask(variants[0])
        ref = GoldenReference(mm_per_px=0.25)
        ref.build(variants, part_mask=mask)
        engine = CalibratedEngine(ref, sensitivity=0.85)
        engine.calibrate_noise_floor(variants, percentile=99.5)

        defect_img = variants[0].copy()
        dcx, dcy = 200, 70
        cv2.circle(defect_img, (dcx, dcy), 8, (0, 0, 0), -1)

        res = engine.inspect(defect_img)
        self.assertIsNotNone(res.heatmap)
        self.assertEqual(res.heatmap.shape, defect_img.shape)

        # The defect must actually be found, at roughly the drawn location
        self.assertGreater(len(res.defects), 0)
        top = max(res.defects, key=lambda d: d.area_px)
        self.assertLess(abs((top.x + top.w / 2) - dcx), 15)
        self.assertLess(abs((top.y + top.h / 2) - dcy), 15)

        diff = np.abs(res.heatmap.astype(np.int32) - defect_img.astype(np.int32)).sum(axis=2)
        defect_region = diff[dcy - 10:dcy + 10, dcx - 10:dcx + 10].mean()
        # Plain light-gray body area, far from the defect, the orange
        # bar, and any part edge.
        clean_region = diff[180:200, 400:420].mean()
        self.assertGreater(
            defect_region, clean_region + 15,
            f"heatmap not visibly hotter at the real defect: "
            f"defect_region={defect_region:.1f} clean_region={clean_region:.1f}",
        )

    def test_heatmap_contract_colorizes_only_where_agreement_is_high(self):
        """
        Direct unit test of _heatmap()'s contract, independent of the
        full detection pipeline: given a per-pixel agreement map, the
        output must be visibly altered from the input specifically where
        agreement is high, and left untouched (exactly equal to the
        input) where agreement is ~0 -- regardless of the underlying
        image's own color (tested here against both a saturated/bright
        region and a flat one, since alpha-blending's raw pixel-value
        shift for the same heat level depends on the base color, which
        is exactly what makes reverse-engineering "hotness" from the
        blended output alone unreliable -- see the end-to-end test above).
        """
        img = self.base_img.copy()
        mask = build_part_mask(img)
        agree = np.zeros((self.h, self.w), dtype=np.float32)
        agree[60:80, 190:210] = 8.0  # inside the orange bar: full 8/8 agreement
        agree[150:170, 250:270] = 8.0  # inside the plain gray body

        heatmap = CalibratedEngine._heatmap(img, agree, 8, mask)
        self.assertEqual(heatmap.shape, img.shape)

        hot_orange = np.abs(heatmap[60:80, 190:210].astype(int) - img[60:80, 190:210].astype(int)).mean()
        hot_gray = np.abs(heatmap[150:170, 250:270].astype(int) - img[150:170, 250:270].astype(int)).mean()
        cold = np.abs(heatmap[210:230, 400:420].astype(int) - img[210:230, 400:420].astype(int)).mean()

        self.assertGreater(hot_orange, cold + 10)
        self.assertGreater(hot_gray, cold + 10)
        self.assertEqual(cold, 0.0)


if __name__ == "__main__":
    unittest.main()
