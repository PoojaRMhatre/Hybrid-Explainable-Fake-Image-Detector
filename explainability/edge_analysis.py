import cv2
import numpy as np

def edge_inconsistency(img_path):
    """
    Robust edge inconsistency analysis.
    Returns score between 0.0 and 10.0
    """

    try:
        # ---------------- LOAD IMAGE ----------------
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Invalid image path or unreadable file")

        # ---------------- STANDARDIZE ----------------
        img = cv2.resize(img, (256, 256))

        # ---------------- PREPROCESS ----------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize lighting
        gray = cv2.equalizeHist(gray)

        # Slight blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---------------- ADAPTIVE CANNY ----------------
        median_val = np.median(blurred)

        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))

        edges = cv2.Canny(blurred, lower, upper)

        # ---------------- EDGE METRICS ----------------
        # Density of edges
        edge_density = np.sum(edges > 0) / edges.size

        # Variance of edges (texture inconsistency)
        edge_variance = np.var(edges)

        # ---------------- SPATIAL CONSISTENCY ----------------
        # Divide into patches and analyze local variation
        patch_size = 32
        h, w = edges.shape
        patch_variances = []

        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = edges[y:y+patch_size, x:x+patch_size]
                if patch.size > 0:
                    patch_variances.append(np.var(patch))

        spatial_variation = np.std(patch_variances)

        # ---------------- FINAL SCORE ----------------
        # Weighted combination (tuned heuristic)
        score_raw = (
            edge_density * 5.0 + 
            (edge_variance / 5000.0) * 3.0 + 
            (spatial_variation / 2000.0) * 2.0
        )

        final_score = np.clip(score_raw, 0, 10)

        return float(round(final_score, 2))

    except Exception as e:
        print(f"[Edge Analysis Error]: {e}")
        return 0.0