import os
import cv2
import numpy as np
from PIL import Image, ImageChops
import tempfile


def ela_score(path, save_dir=None, quality=90):
    """
    Advanced Error Level Analysis (ELA) with smooth heatmap.
    
    Returns:
        score (0–10)
        heatmap filename (if saved)
    """

    temp_file = None

    try:
        if not os.path.exists(path):
            raise ValueError("Image path does not exist")

        # ---------------- LOAD IMAGE ----------------
        original = Image.open(path)

        if original.mode != "RGB":
            original = original.convert("RGB")

        original = original.resize((256, 256))

        # ---------------- TEMP JPEG ----------------
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_file = tmp.name

        original.save(temp_file, "JPEG", quality=quality)
        resaved = Image.open(temp_file)

        # ---------------- ELA COMPUTATION ----------------
        ela_image = ImageChops.difference(original, resaved)
        ela_array = np.array(ela_image).astype(np.float32)

        # ---------------- SCORING ----------------
        avg_diff = np.mean(ela_array)
        std_dev = np.std(ela_array)
        max_diff = np.max(ela_array)

        score_raw = (avg_diff * 0.5) + (std_dev * 1.5) + (max_diff * 0.2)
        score = np.clip(score_raw / 12.0, 0, 10)

        # ---------------- HEATMAP ----------------
        heatmap_filename = None

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Convert to grayscale
            ela_gray = cv2.cvtColor(ela_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)

            # ---------------- LOG SCALING ----------------
            ela_log = np.log1p(ela_gray.astype(np.float32))

            # ---------------- PERCENTILE NORMALIZATION ----------------
            p1, p99 = np.percentile(ela_log, (1, 99))
            ela_clipped = np.clip(ela_log, p1, p99)

            ela_norm = cv2.normalize(
                ela_clipped, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # ---------------- EDGE-PRESERVING SMOOTHING ----------------
            ela_smooth = cv2.bilateralFilter(
                ela_norm,
                d=9,              # neighborhood size
                sigmaColor=75,    # color smoothing
                sigmaSpace=75     # spatial smoothing
            )

            # ---------------- EXTRA SOFT BLUR ----------------
            ela_smooth = cv2.GaussianBlur(ela_smooth, (5, 5), 0)

            # ---------------- CONTRAST BOOST ----------------
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            ela_enhanced = clahe.apply(ela_smooth)

            # ---------------- UPSCALE FOR BETTER VISUALS ----------------
            ela_upscaled = cv2.resize(
                ela_enhanced,
                (512, 512),
                interpolation=cv2.INTER_CUBIC
            )

            # ---------------- COLOR MAP ----------------
            heatmap = cv2.applyColorMap(ela_upscaled, cv2.COLORMAP_JET)

            # ---------------- SAVE ----------------
            filename = os.path.basename(path)
            heatmap_filename = f"ela_{filename}"
            heatmap_path = os.path.join(save_dir, heatmap_filename)

            cv2.imwrite(heatmap_path, heatmap)

        return float(round(score, 2)), heatmap_filename

    except Exception as e:
        print(f"[ELA Error]: {str(e)}")
        return 0.0, None

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass