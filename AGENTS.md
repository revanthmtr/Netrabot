# AGENTS.md — Visual Inspection Engine

Read this before changing any detection code. These decisions came from
validation on real defect samples, not from theory. Violating them
reintroduces bugs that were already found and fixed.

## What this project is

Automated optical inspection (AOI) for printed safety labels. A part is
photographed in a fixed camera rig and compared against a statistical
golden reference. Output: PASS/FAIL plus located, classified defects.

Domain priority: **a missed defect is unacceptable; a false alarm is
merely expensive.** All tuning decisions must favour recall.

## Hard architectural rules

### 1. NEVER use homography/perspective warp for registration

`cv2.findHomography` + `cv2.warpPerspective` silently absorbs genuine
position and rotation defects — it cannot distinguish "camera was
tilted" from "part is mounted crooked". This was an actual bug found in
development.

Use `cv2.phaseCorrelate` for sub-pixel TRANSLATION only. Shift beyond
tolerance is REPORTED as a defect, never corrected away.

### 2. Two detection tracks, OR-fused. Neither alone is sufficient.

Validated on four known-defect samples:

| Track | Method | Catches | Proven on |
|---|---|---|---|
| Localized | 9 detectors → union → noise-calibrated → saliency-ranked | specks, scratches, ink bridges | samples A, B, C |
| Regional | per-ROI aggregate ΔE / density / texture / topology | fade, smear, density loss | sample D |

Sample D (faded panel) is invisible to blob detection — a uniform
regional shift has no edge or local contrast anomaly to latch onto.
Ranking by blob saliency buried it at rank 196/412. The regional check
separated it cleanly: ΔE 12.69 vs 0.49 on good parts, tolerance 3.0.

A part FAILS if EITHER track fails.

### 3. Golden reference must be built from 20–30 real good parts

Not one master image. Per-pixel `std` is what distinguishes "this pixel
naturally varies here" from "this is a defect". Edges vary a lot; flat
areas vary little.

With one sample the code falls back to spatial variance estimation. It
works but is materially weaker — validation showed ~300–590 candidate
blobs per part instead of a handful. **The golden set is the
load-bearing input; the engine's accuracy ceiling is set by it.**

### 4. Thresholds are MEASURED, never guessed

Run detectors on known-good parts, take the p99.5 response — that is the
noise floor. Set thresholds just above it (`margin` ≈ 1.02–1.22).

Hardcoding a threshold like `if ssim > 0.99` is the single biggest
failure mode here. It was tried; it produced plausible numbers that were
wrong, and passed samples with real defects.

### 5. Never DROP candidates — rank them

Zero-miss requires keeping every candidate for audit. Usability requires
the operator not seeing 590 boxes. Resolve this by ranking (saliency
score), not filtering. Full list persists to the database.

Saliency weights (validated): 0.45 detector agreement, 0.25 log-scaled
area, 0.20 peak deviation, 0.10 compactness.

### 6. Colour comparison uses CIE Lab + ΔE, never raw RGB

RGB distance does not match perceived colour difference and cannot be
written into a defensible tolerance spec. Lab/ΔE can.

## Validation requirement

Any change to detection logic must be re-run against the labelled sample
set. Report per-defect rank before and after. A change that improves one
sample's rank while worsening another's is NOT an improvement — check
both tracks.

Never claim accuracy without measured false-accept / false-reject rates
on a labelled set.

## Stack

OpenCV, scikit-image (SSIM, Lab colour), NumPy. Optional phase 2:
`anomalib` (PatchCore/PaDiM) for unknown defect types — only after the
rule-based system is stable and a labelled dataset exists.
