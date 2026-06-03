import os
import json
import logging
import base64
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# NVIDIA NIM API details
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") 

def encode_image(image_path):
    try:
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize for optimization (Gemma 4 handles variable resolutions well, 
        # but 800x800 keeps payload sizes manageable)
        img.thumbnail((512, 512))

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    except Exception as e:
        logging.error(f"IMAGE ENCODING FAILED: {str(e)}")
        return None

def get_verdict_and_explanation(image_path, ela=0, noise=0, exif="", freq=0, edge=0):
    try:
        # Encode image
        image_base64 = encode_image(image_path)

        if not image_base64:
            return "Authentic", 0.0, "**ERROR:** Image processing failed."

        # The core instructions (No longer need to paste base64 text here)
        text_prompt = """
You are a highly rigid, literal digital forensics AI.
Analyze the provided image conceptually and visually.

Follow STRICT rules:

CLASSIFICATION RULES (EVALUATE IN ORDER):
1. TEXT OVERLAY OVERRIDE:
If any visible text, meme, watermark, UI → "Manipulated / Edited"

2. AI GENERATION ARTIFACTS:
If weird anatomy, extra fingers, plastic texture → "AI Generated"

3. NATURAL PHOTO:
Otherwise → "Authentic"

CRITICAL OUTPUT:
Return ONLY JSON:
{
  "verdict": "AI Generated | Manipulated / Edited | Authentic",
  "confidence": number (50.0–99.9),
  "explanation": [
    "bullet 1",
    "bullet 2",
    "bullet 3"
  ]
}
"""

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json"
        }

        # Structure the payload for a true Multimodal model
        payload = {
            "model": "google/gemma-4-31b-it",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.2, 
            "top_p": 0.95,
            # Enable Gemma 4's built-in reasoning to better process the strict forensic rules
            "chat_template_kwargs": {"enable_thinking": False}
        }

        # Make the API call
        response = requests.post(INVOKE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        text_output = response_data['choices'][0]['message']['content'].strip()

        # Try parsing JSON safely, stripping out markdown formatting if present
        try:
            data = json.loads(text_output)
        except json.JSONDecodeError:
            text_output = text_output.split("```json")[-1].split("```")[0].strip()
            data = json.loads(text_output)

        verdict = data.get('verdict', 'Authentic')
        confidence = float(data.get('confidence', 60.0))
        explanation = data.get('explanation', [
            "Visual analysis attempted",
            "Limited image understanding",
            "Confidence reduced"
        ])

        logging.info(f"Gemma 4 Processed. Verdict: {verdict}")
        return verdict, confidence, explanation

    except Exception as e:
        error_msg = str(e)
        logging.error(f"GEMMA NIM CRASHED: {error_msg}")
        return "Authentic", 0.0, f"**API ERROR:** {error_msg}"