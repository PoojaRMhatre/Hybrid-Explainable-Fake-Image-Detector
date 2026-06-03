import cv2
import numpy as np

def noise_variance(path):
    """
    Advanced Noise Variance Estimation using high-pass filtering.

    Returns:
        float: score (0.0 - 10.0)
    """

    try:
        # ---------------- LOAD ----------------
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Invalid image path")

        # ---------------- STANDARDIZE ----------------
        img = cv2.resize(img, (256, 256))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize lighting
        gray = cv2.equalizeHist(gray)

        # ---------------- NOISE EXTRACTION ----------------
        # Immerkær-like kernel (high-pass filter)
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ], dtype=np.float32)

        noise_map = cv2.filter2D(gray.astype(np.float32), -1, kernel)

        # ---------------- METRICS ----------------
        variance = np.var(noise_map)
        std_dev = np.std(noise_map)

        # Spatial consistency (important)
        patch_size = 32
        h, w = noise_map.shape
        patch_vars = []

        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = noise_map[y:y+patch_size, x:x+patch_size]
                if patch.size > 0:
                    patch_vars.append(np.var(patch))

        spatial_variation = np.std(patch_vars)

        # ---------------- SCORING ----------------
        # AI images often:
        # - have overly smooth noise
        # - or inconsistent synthetic noise

        score_raw = (
            (variance / 5000.0) * 4.0 +
            (std_dev / 100.0) * 3.0 +
            (spatial_variation / 2000.0) * 3.0
        )

        score = np.clip(score_raw, 0, 10)

        return float(round(score, 2))

    except Exception as e:
        print(f"[Noise Analysis Error]: {str(e)}")
        return 0.0