"""
I/O and preprocessing utilities for the AOI Inspection Engine.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def load_image(path: Union[str, Path], max_dim: int = 1600) -> np.ndarray:
    """
    Load an image from disk, transparently handling .HEIC, .heic, .png, .jpg, etc.
    Optionally scales down images exceeding max_dim for fast real-time inspection.

    Args:
        path: Path to the image file.
    Returns:
        np.ndarray: BGR uint8 image array compatible with OpenCV.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found at: {path}")

    ext = path_obj.suffix.lower()
    if ext in ('.heic', '.heif'):
        import pillow_heif
        from PIL import Image, ImageOps

        heif_file = pillow_heif.read_heif(str(path_obj))
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        # Apply EXIF orientation if present
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        rgb = np.array(image)
        if len(rgb.shape) == 2:
            img = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif rgb.shape[2] == 4:
            img = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image from {path} with cv2.imread")

    if max_dim is not None and max_dim > 0:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def build_part_mask(img: np.ndarray) -> np.ndarray:
    """
    Isolate the part from the background via:
    threshold + morphological close + largest external contour + erode 10px.

    Args:
        img: BGR or grayscale numpy image array.
    Returns:
        np.ndarray: Single-channel uint8 mask (0 or 255) of shape (H, W).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    img_area = h * w

    # Step 1: Threshold (Otsu)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Morphological close
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    # Step 3: Largest external contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=np.uint8) * 255

    largest = max(contours, key=cv2.contourArea)

    # If background was white and inverted, recheck if contour encompasses full image (>98%)
    if cv2.contourArea(largest) > 0.98 * img_area:
        thresh_inv = cv2.bitwise_not(thresh)
        closed_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, close_kernel)
        contours_inv, _ = cv2.findContours(closed_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_inv:
            largest = max(contours_inv, key=cv2.contourArea)

    raw_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(raw_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Step 4: Erode 10px
    erode_kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(raw_mask, erode_kernel, iterations=10)

    return eroded_mask
