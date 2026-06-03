import cv2
import numpy as np

def frequency_analysis(img_path):
    """
    Robust frequency-domain analysis for detecting synthetic artifacts.
    Returns a score between 0.0 and 10.0
    """

    try:
        # ---------------- LOAD IMAGE ----------------
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image path or unreadable file")

        # ---------------- STANDARDIZE SIZE ----------------
        img = cv2.resize(img, (256, 256))  # Normalize input size

        # ---------------- NORMALIZE INTENSITY ----------------
        img = cv2.equalizeHist(img)

        # ---------------- FFT ----------------
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        magnitude = np.log1p(np.abs(fshift))  # log(1+x) is safer

        # ---------------- FREQUENCY SPLIT ----------------
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Dynamic radius (5% of image size)
        r = int(min(h, w) * 0.05)

        center_region = magnitude[cy - r:cy + r, cx - r:cx + r]

        # Mask for high-frequency region
        mask = np.ones_like(magnitude, dtype=bool)
        mask[cy - r:cy + r, cx - r:cx + r] = False

        high_freq_region = magnitude[mask]

        # ---------------- METRICS ----------------
        avg_high = np.mean(high_freq_region)
        avg_low = np.mean(center_region)

        std_high = np.std(high_freq_region)

        # Ratio is more stable than difference
        ratio = avg_high / (avg_low + 1e-6)

        # Combine metrics (tuned heuristic)
        score_raw = (ratio * 2.0) + (std_high * 0.5)

        # ---------------- NORMALIZATION ----------------
        score = np.clip(score_raw, 0, 10)

        return float(round(score, 2))

    except Exception as e:
        print(f"[Frequency Analysis Error]: {e}")
        return 0.0