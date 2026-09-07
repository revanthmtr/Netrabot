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


if __name__ == "__main__":
    unittest.main()
