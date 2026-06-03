from PIL import Image, ExifTags

def exif_analysis(path):
    """
    Advanced EXIF forensic analysis.

    Returns:
        dict:
        {
            "score": float (0–10),
            "label": str,
            "details": dict
        }
    """

    try:
        img = Image.open(path)
        exif_raw = img.getexif()

        # ---------------- NO EXIF ----------------
        if not exif_raw:
            return {
                "score": 7.5,
                "label": "No EXIF Metadata",
                "details": {"reason": "Stripped metadata or AI-generated"}
            }

        # ---------------- PARSE EXIF ----------------
        exif_data = {}
        for tag_id, value in exif_raw.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif_data[tag] = value

        # ---------------- EXTRACT FIELDS ----------------
        software = str(exif_data.get("Software", "")).lower()
        processing = str(exif_data.get("ProcessingSoftware", "")).lower()
        make = str(exif_data.get("Make", "")).lower()
        model = str(exif_data.get("Model", "")).lower()

        combined_software = software + " " + processing

        # ---------------- DETECTION RULES ----------------
        ai_tools = [
            "midjourney", "dall-e", "stable diffusion",
            "comfyui", "automatic1111"
        ]

        editing_tools = [
            "photoshop", "gimp", "lightroom", "snapseed", "canva"
        ]

        score = 0.0
        flags = []

        # ---- AI SOFTWARE DETECTED ----
        for tool in ai_tools:
            if tool in combined_software:
                score = 9.5
                flags.append(f"AI Tool Detected: {tool}")

        # ---- EDITING SOFTWARE ----
        for tool in editing_tools:
            if tool in combined_software:
                score = max(score, 7.0)
                flags.append(f"Editing Software: {tool}")

        # ---- CAMERA CHECK ----
        has_camera = any([
            "Make" in exif_data,
            "Model" in exif_data,
            "ExposureTime" in exif_data,
            "FNumber" in exif_data
        ])

        if has_camera:
            score = min(score, 3.0)  # more likely real
        else:
            score = max(score, 6.5)
            flags.append("No Camera Hardware Data")

        # ---- INCONSISTENCY CHECK ----
        if has_camera and combined_software:
            score = max(score, 8.0)
            flags.append("Camera + Editing Software Mismatch")

        # ---- FINAL LABEL ----
        if score >= 8:
            label = "Likely AI / Manipulated"
        elif score >= 5:
            label = "Suspicious Metadata"
        else:
            label = "Likely Authentic"

        return {
            "score": round(score, 2),
            "label": label,
            "details": {
                "make": make,
                "model": model,
                "software": combined_software.strip(),
                "flags": flags
            }
        }

    except Exception as e:
        return {
            "score": 5.0,
            "label": "EXIF Read Error",
            "details": {"error": str(e)}
        }
exif_flag = exif_analysis