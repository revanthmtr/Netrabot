#!/usr/bin/env python3
"""
Multi-Model Preview Dashboard Server for NETRABOT AOI Visual Inspection.

Structure:
  data/models/<model_name>/golden/   — golden reference images for this model
  data/models/<model_name>/samples/  — sample images to inspect against this model
  data/models/<model_name>/spec.json — part spec for this model
  data/models/<model_name>/output/   — inspection results (annotated images, reports)
"""

import cgi
import json
import os
import shutil
import subprocess
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "data" / "models"
PORT = 8090

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".bmp", ".tiff", ".tif"}


def ensure_model_dirs(model_name):
    base = MODELS_DIR / model_name
    (base / "golden").mkdir(parents=True, exist_ok=True)
    (base / "samples").mkdir(parents=True, exist_ok=True)
    (base / "output" / "annotated").mkdir(parents=True, exist_ok=True)
    (base / "output" / "reports").mkdir(parents=True, exist_ok=True)
    return base


def list_images(directory):
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(
        [
            f.name
            for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ]
    )


class InspectionDashboardHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def log_message(self, format, *args):
        # quieter logging
        pass

    # ── GET ──────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/dashboard/index.html"
            return super().do_GET()

        # GET /api/models — list all models with summary
        if self.path == "/api/models":
            return self._json_response(self._get_models_list())

        # GET /api/models/<name> — full data for one model
        if self.path.startswith("/api/models/") and self.path.count("/") == 3:
            model_name = self.path.split("/")[3]
            return self._json_response(self._get_model_data(model_name))

        # Dynamic HEIC to PNG transcoding for browser compatibility
        clean_path = self.path.split("?")[0].lstrip("/")
        if clean_path.lower().endswith((".heic", ".heif")):
            filepath = ROOT_DIR / clean_path
            if filepath.exists() and filepath.is_file():
                try:
                    import cv2
                    from engine.io_utils import load_image
                    img = load_image(filepath)
                    success, buffer = cv2.imencode(".png", img)
                    if success:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Content-Length", str(len(buffer)))
                        self.send_header("Cache-Control", "public, max-age=3600")
                        self.end_headers()
                        self.wfile.write(buffer.tobytes())
                        return
                except Exception as e:
                    pass

        # Serve static files (images, CSS, JS)
        return super().do_GET()

    # ── POST ─────────────────────────────────────────────────────────
    def do_POST(self):
        # POST /api/models — create a new model
        if self.path == "/api/models":
            body = self._read_json_body()
            if not body or "name" not in body:
                return self._json_response({"error": "name required"}, 400)
            name = body["name"].strip().replace(" ", "-")
            if not name:
                return self._json_response({"error": "name cannot be empty"}, 400)
            base = ensure_model_dirs(name)
            # Copy default spec if not existing
            spec_path = base / "spec.json"
            if not spec_path.exists():
                default_spec = ROOT_DIR / "specs" / "part_spec.json"
                if default_spec.exists():
                    shutil.copy2(default_spec, spec_path)
                else:
                    spec_path.write_text(json.dumps({"part_id": name}, indent=2))
            return self._json_response({"status": "created", "model": name}, 201)

        # POST /api/models/<name>/upload/golden — upload golden images
        if self.path.endswith("/upload/golden"):
            parts = self.path.split("/")
            model_name = parts[3]
            return self._handle_upload(model_name, "golden")

        # POST /api/models/<name>/upload/samples — upload sample images
        if self.path.endswith("/upload/samples"):
            parts = self.path.split("/")
            model_name = parts[3]
            return self._handle_upload(model_name, "samples")

        # POST /api/models/<name>/inspect — run inspection for a model
        if self.path.endswith("/inspect"):
            parts = self.path.split("/")
            model_name = parts[3]
            return self._run_inspection(model_name)

        return self._json_response({"error": "not found"}, 404)

    # ── DELETE ───────────────────────────────────────────────────────
    def do_DELETE(self):
        # DELETE /api/models/<name> — delete a model
        if self.path.startswith("/api/models/") and self.path.count("/") == 3:
            model_name = self.path.split("/")[3]
            model_dir = MODELS_DIR / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                return self._json_response({"status": "deleted", "model": model_name})
            return self._json_response({"error": "model not found"}, 404)

        # DELETE /api/models/<name>/golden/<filename> — delete a golden image
        if "/golden/" in self.path:
            parts = self.path.split("/")
            model_name = parts[3]
            filename = parts[5]
            fp = MODELS_DIR / model_name / "golden" / filename
            if fp.exists():
                fp.unlink()
                return self._json_response({"status": "deleted"})
            return self._json_response({"error": "file not found"}, 404)

        # DELETE /api/models/<name>/samples/<filename> — delete a sample image
        if "/samples/" in self.path:
            parts = self.path.split("/")
            model_name = parts[3]
            filename = parts[5]
            fp = MODELS_DIR / model_name / "samples" / filename
            if fp.exists():
                fp.unlink()
                stem = fp.stem
                out = MODELS_DIR / model_name / "output"
                for p in [
                    out / "reports" / f"{stem}_report.json",
                    out / "annotated" / f"{stem}_annotated.png",
                    out / "annotated" / f"{stem}_heatmap.png"
                ]:
                    if p.exists():
                        p.unlink()
                return self._json_response({"status": "deleted"})
            return self._json_response({"error": "file not found"}, 404)

        return self._json_response({"error": "not found"}, 404)

    # ── Helpers ──────────────────────────────────────────────────────
    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            return json.loads(raw)
        except Exception:
            return None

    def _get_models_list(self):
        models = []
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            return {"models": []}
        for d in sorted(MODELS_DIR.iterdir()):
            if not d.is_dir():
                continue
            golden_imgs = list_images(d / "golden")
            sample_imgs = list_images(d / "samples")
            spec = {}
            spec_path = d / "spec.json"
            if spec_path.exists():
                try:
                    with open(spec_path) as f:
                        spec = json.load(f)
                except Exception:
                    pass
            # Count reports
            reports_dir = d / "output" / "reports"
            report_count = len(list(reports_dir.glob("*_report.json"))) if reports_dir.exists() else 0
            models.append({
                "name": d.name,
                "part_id": spec.get("part_id", d.name),
                "description": spec.get("description", ""),
                "golden_count": len(golden_imgs),
                "sample_count": len(sample_imgs),
                "report_count": report_count,
                "golden_images": golden_imgs,
                "thumbnail_url": f"/data/models/{d.name}/golden/{golden_imgs[0]}" if golden_imgs else None,
            })
        return {"models": models}

    def _get_model_data(self, model_name):
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            return {"error": "model not found"}

        spec = {}
        spec_path = model_dir / "spec.json"
        if spec_path.exists():
            try:
                with open(spec_path) as f:
                    spec = json.load(f)
            except Exception:
                pass

        golden_imgs = list_images(model_dir / "golden")
        sample_imgs = list_images(model_dir / "samples")

        # Load reports
        reports_dir = model_dir / "output" / "reports"
        parts = []
        if reports_dir.exists():
            for rpt_file in sorted(reports_dir.glob("*_report.json")):
                try:
                    with open(rpt_file) as f:
                        rep = json.load(f)
                    stem = rpt_file.stem.replace("_report", "")
                    parts.append({
                        "report": rep,
                        "sample_image_url": f"/data/models/{model_name}/samples/{rep.get('file', stem + '.png')}",
                        "annotated_image_url": f"/data/models/{model_name}/output/annotated/{stem}_annotated.png",
                        "heatmap_image_url": f"/data/models/{model_name}/output/annotated/{stem}_heatmap.png",
                    })
                except Exception:
                    pass

        golden_ref_path = model_dir / "output" / "golden_reference.png"
        return {
            "model_name": model_name,
            "spec": spec,
            "golden_images": golden_imgs,
            "golden_image_urls": [f"/data/models/{model_name}/golden/{g}" for g in golden_imgs],
            "golden_reference_url": f"/data/models/{model_name}/output/golden_reference.png" if golden_ref_path.exists() else None,
            "sample_images": sample_imgs,
            "parts": parts,
        }

    def _handle_upload(self, model_name, subfolder):
        base = ensure_model_dirs(model_name)
        target_dir = base / subfolder

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )
            saved = []
            file_items = form["files"] if "files" in form else []
            if not isinstance(file_items, list):
                file_items = [file_items]
            for item in file_items:
                if item.filename:
                    fname = Path(item.filename).name
                    dest = target_dir / fname
                    with open(dest, "wb") as f:
                        f.write(item.file.read())
                    saved.append(fname)
            return self._json_response({
                "status": "uploaded",
                "model": model_name,
                "subfolder": subfolder,
                "files": saved,
                "total": len(list_images(target_dir)),
            })
        else:
            return self._json_response({"error": "multipart/form-data required"}, 400)

    def _run_inspection(self, model_name):
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            return self._json_response({"error": "model not found"}, 404)

        golden_dir = model_dir / "golden"
        samples_dir = model_dir / "samples"
        output_dir = model_dir / "output"
        spec_path = model_dir / "spec.json"

        if not list_images(golden_dir):
            return self._json_response({"error": "No golden images. Upload at least 1 (20-30 recommended)."}, 400)
        if not list_images(samples_dir):
            return self._json_response({"error": "No sample images to inspect."}, 400)

        # Use project spec if model-specific spec doesn't exist
        if not spec_path.exists():
            spec_path = ROOT_DIR / "specs" / "part_spec.json"

        try:
            # Build golden reference image
            build_ref_script = f"""
import cv2, sys
sys.path.insert(0, '{ROOT_DIR}')
from engine.io_utils import load_image, build_part_mask
from engine.inspection_engine import GoldenReference
from pathlib import Path
import json

golden_dir = Path('{golden_dir}')
output_dir = Path('{output_dir}')
output_dir.mkdir(parents=True, exist_ok=True)

exts = {{'.png','.jpg','.jpeg','.heic','.heif','.bmp','.tiff','.tif'}}
golden_files = sorted(p for p in golden_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
images = [load_image(f) for f in golden_files]
mask = build_part_mask(images[0])

with open('{spec_path}') as f:
    spec = json.load(f)
mm = float(spec.get('calibration', {{}}).get('mm_per_px', 1.0))
ref = GoldenReference(mm_per_px=mm)
ref.build(images, part_mask=mask)
cv2.imwrite(str(output_dir / 'golden_reference.png'), ref.golden_bgr)
print('Built golden reference')
"""
            subprocess.run(
                [sys.executable, "-c", build_ref_script],
                check=True,
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
            )

            # Run the inspection CLI
            cmd = [
                sys.executable,
                str(ROOT_DIR / "run_inspection.py"),
                "--golden", str(golden_dir),
                "--input", str(samples_dir),
                "--spec", str(spec_path),
                "--out", str(output_dir),
                "--top-n", "15",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
            if result.returncode != 0:
                return self._json_response({
                    "error": f"Inspection failed: {result.stderr or result.stdout}"
                }, 500)

            # Generate heatmaps
            heatmap_script = f"""
import cv2, sys, numpy as np
sys.path.insert(0, '{ROOT_DIR}')
from pathlib import Path
from engine.io_utils import load_image
golden = cv2.imread('{output_dir / "golden_reference.png"}')
if golden is not None:
    g_gray = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    exts = {{'.png','.jpg','.jpeg','.heic','.heif','.bmp','.tiff','.tif'}}
    for sf in Path('{samples_dir}').iterdir():
        if sf.suffix.lower() in exts:
            try:
                sample = load_image(sf)
                if sample is not None:
                    if sample.shape[:2] != g_gray.shape:
                        res_m = cv2.matchTemplate(g_gray, cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                        _, _, _, max_loc = cv2.minMaxLoc(res_m)
                        canvas = np.copy(golden)
                        canvas[max_loc[1]:max_loc[1]+sample.shape[0], max_loc[0]:max_loc[0]+sample.shape[1]] = sample
                        sample = canvas
                    s_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(s_gray, g_gray)
                    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                    cv2.imwrite(str(Path('{output_dir / "annotated"}') / f'{{sf.stem}}_heatmap.png'), heatmap)
            except Exception:
                pass
"""
            subprocess.run(
                [sys.executable, "-c", heatmap_script],
                check=True,
                cwd=str(ROOT_DIR),
                capture_output=True,
            )

            # Generate evidence crops
            crop_script = f"""
import json, sys, cv2, numpy as np
sys.path.insert(0, '{ROOT_DIR}')
from pathlib import Path
from engine.io_utils import load_image
from engine.defect_crops import generate_all_crops
golden = load_image('{output_dir / "golden_reference.png"}')
if golden is not None:
    with open('{spec_path}') as sf:
        sp = json.load(sf)
    mm = float(sp.get('calibration', {{}}).get('mm_per_px', 1.0))
    for rpt_path in (Path('{output_dir}') / 'reports').glob('*_report.json'):
        try:
            with open(rpt_path) as f:
                rpt = json.load(f)
            s_file = Path('{samples_dir}') / rpt.get('file', '')
            if s_file.exists():
                s_img = load_image(s_file)
                if s_img.shape[:2] != golden.shape[:2]:
                    res_m = cv2.matchTemplate(cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY), cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(res_m)
                    canvas = np.copy(golden)
                    canvas[max_loc[1]:max_loc[1]+s_img.shape[0], max_loc[0]:max_loc[0]+s_img.shape[1]] = s_img
                    s_img = canvas
                generate_all_crops(golden, s_img, rpt.get('defects', []), '{output_dir}', rpt_path.stem.replace('_report', ''), mm_per_px=mm, max_crops=15)
                with open(rpt_path, 'w') as f:
                    json.dump(rpt, f, indent=2)
        except Exception:
            pass
"""
            subprocess.run([sys.executable, "-c", crop_script], cwd=str(ROOT_DIR))

            return self._json_response({
                "status": "success",
                "model": model_name,
                "stdout": result.stdout[-2000:] if result.stdout else "",
            })
        except Exception as e:
            return self._json_response({
                "error": str(e),
                "traceback": traceback.format_exc()[-1500:],
            }, 500)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), InspectionDashboardHandler)
    print(f"NETRABOT AOI Dashboard → http://localhost:{PORT}")
    print(f"Models directory: {MODELS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
