import os
import time
import json
import logging
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
import markdown

# Email and Token Security
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature

# --- CUSTOM PYTORCH IMPORTS (Left intact for fallback initialization, but bypassed in logic) ---
import torch
import torch.nn as nn
from torchvision import models, transforms

# Custom Explainability Modules
from explainability.ela_analysis import ela_score
from explainability.noise_analysis import noise_variance
from explainability.exif_check import exif_flag
from explainability.frequency_analysis import frequency_analysis
from explainability.edge_analysis import edge_inconsistency

# Load Environment Variables & Logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_secret_key_aletheia_2026") 
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'exr', 'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.secret_key)

try:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["aletheia_forensics"]
    users_col = db["users"]
    scans_col = db["scan_history"]
    client.server_info()
except Exception as e:
    logging.critical(f"Database Connection Failed: {e}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_model = None
class_mapping = {}

try:
    logging.info(f"Booting PyTorch Inference Engine on {device}...")
    model_path = "models/best_model.pth"
    mapping_path = "models/class_indices.json"
    
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            loaded_mapping = json.load(f)
            class_mapping = {int(k): str(v) for k, v in loaded_mapping.items()} if all(isinstance(k, str) and k.isdigit() for k in loaded_mapping.keys()) else {int(v): str(k) for k, v in loaded_mapping.items()}
    else:
        class_mapping = {0: "AI Generated", 1: "Authentic", 2: "Manipulated / Edited"}
        
    num_classes = len(class_mapping)
    custom_model = models.efficientnet_b3(weights=None)
    num_ftrs = custom_model.classifier[1].in_features
    custom_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        custom_model.load_state_dict(torch.load(model_path, map_location=device))
        custom_model.to(device)
        custom_model.eval() 
        logging.info("✅ Custom PyTorch EfficientNet Model Loaded Successfully!")
    else:
        logging.error("❌ Model weights not found. Running in blind fallback mode.")
        custom_model = None
except Exception as e:
    logging.error(f"Error loading custom PyTorch model: {e}")
    custom_model = None

def purge_old_uploads(folder_path, max_hours=24):
    try:
        now = time.time()
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > (max_hours * 3600):
                try: os.remove(file_path)
                except: pass
    except: pass

def handle_pdf(pdf_path, filename):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=150) 
        image_filename = f"{filename.rsplit('.', 1)[0]}_extracted.jpg"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
        pix.save(image_path)
        doc.close()
        return image_filename, image_path
    except Exception as e: 
        logging.error(f"PDF Extraction Error: {e}")
        return None, None

def send_async_email(to, subject, html_body):
    try:
        msg = Message(subject, recipients=[to], html=html_body, sender=app.config['MAIL_USERNAME'])
        mail.send(msg)
        return True
    except Exception as e:
        return False

# =====================================================================
# AI-LED DECISION ENGINE (PURE GEMINI VISION PARADIGM)
# =====================================================================
def forensic_predict(img_path):
    # 1. Telemetry Extraction (STRICTLY FOR UI DISPLAY ONLY)
    ela_value, heatmap_name = ela_score(img_path, app.config["UPLOAD_FOLDER"])
    noise_value = noise_variance(img_path)
    exif_status = exif_flag(img_path)
    freq_value = frequency_analysis(img_path)
    edge_value = edge_inconsistency(img_path)

    # 2. Gemini defines the absolute truth (Verdict + Explanation combined)
    from explainability.gemini_explainer import get_verdict_and_explanation
    gemini_label, gemini_conf, raw_explanation = get_verdict_and_explanation(
        img_path, ela_value, noise_value, exif_status, freq_value, edge_value
    )

    final_label = gemini_label
    final_conf = round(max(50.0, min(99.9, gemini_conf)), 2)

    return final_label, final_conf, ela_value, noise_value, exif_status, freq_value, edge_value, heatmap_name, raw_explanation

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =====================================================================
# ROUTES & AUTHENTICATION
# =====================================================================
@app.route("/")
def home(): 
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name").strip()
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")
        institution = request.form.get("institution", "").strip() 

        if users_col.find_one({"email": email}):
            flash("Email is already registered. Please log in.", "error")
            return redirect(url_for("register"))

        users_col.insert_one({
            "name": name, 
            "email": email, 
            "password_hash": generate_password_hash(password),
            "institution": institution,
            "verified": False, 
            "created_at": datetime.now(timezone.utc)
        })
        
        token = serializer.dumps(email, salt='email-confirm')
        link = url_for('verify_email', token=token, _external=True)
        html_body = f'<h3>Welcome!</h3><p>Click below to verify your account:</p><a href="{link}">Verify Account</a>'
        send_async_email(email, "Verify Account - Aletheia", html_body)
        
        flash("Registration successful! Please check your email.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route('/verify_email/<token>')
def verify_email(token):
    try: 
        email = serializer.loads(token, salt='email-confirm', max_age=1800)
    except: 
        flash("Link expired or invalid. Please request a new one.", "error")
        return redirect(url_for('login'))
        
    users_col.update_one({'email': email}, {'$set': {'verified': True}})
    flash('Account verified! You can now log in.', 'success')
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")
        user = users_col.find_one({"email": email})
        
        if user and check_password_hash(user["password_hash"], password):
            if not user.get('verified', False):
                flash("Please check your email to verify your account.", "warning")
                return redirect(url_for("login"))
                
            session["user_email"] = user["email"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Successfully logged out.", "success")
    return redirect(url_for("home"))

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        user = users_col.find_one({"email": email})
        if user:
            token = serializer.dumps(email, salt='password-reset')
            link = url_for('reset_password', token=token, _external=True)
            html_body = f'<p>Click to reset your password:</p><a href="{link}">Reset Password</a>'
            send_async_email(email, "Reset Password Request", html_body)
        flash("If registered, a link has been sent.", "success")
        return redirect(url_for("login"))
    return render_template("forgot_password.html")

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try: 
        email = serializer.loads(token, salt='password-reset', max_age=1800)
    except: 
        flash("Reset link is invalid or expired.", "error")
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = generate_password_hash(request.form.get('password'))
        users_col.update_one({'email': email}, {'$set': {'password_hash': new_password}})
        flash('Password successfully updated!', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', token=token)

@app.route("/dashboard")
def dashboard():
    if not session.get("user_email"): return redirect(url_for("login"))
    
    # 1. Fetch from MongoDB (which returns the time in UTC)
    scans = list(scans_col.find({"user_email": session["user_email"]}).sort("timestamp", -1))
    
    # 2. TIMEZONE FIX: Convert MongoDB's UTC time to your laptop's exact local time
    for scan in scans:
        if "timestamp" in scan and isinstance(scan["timestamp"], datetime):
            # Tags the time as UTC, converts it to your OS local time, and strips the tag for the HTML template
            scan["timestamp"] = scan["timestamp"].replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)

    # 3. Calculate stats and render
    ai_detected = sum(1 for s in scans if s['prediction'] in ['AI Generated', 'Manipulated / Edited'])
    return render_template("dashboard.html", scans=scans, total=len(scans), ai_count=ai_detected, authentic_count=len(scans)-ai_detected)

@app.route("/report/<scan_id>")
def view_report(scan_id):
    if not session.get("user_email"): return redirect(url_for("login"))
    try:
        scan_data = scans_col.find_one({"_id": ObjectId(scan_id), "user_email": session.get("user_email")})
        if not scan_data:
            flash("Report not found.", "error")
            return redirect(url_for("dashboard"))
            
        return render_template("result.html", 
                               label=scan_data.get("prediction", "Unknown"), prob=scan_data.get("confidence", 0.0),
                               ela=scan_data.get("ela_score", 0.0), noise=scan_data.get("noise_variance", 0.0), 
                               exif=scan_data.get("exif_status", "Clean"), freq=scan_data.get("freq_score", 0.0), 
                               edge=scan_data.get("edge_score", 0.0), explanation=scan_data.get("explanation", ""), 
                               image=scan_data.get("filename"), heatmap=scan_data.get("heatmap_filename"))
    except: return redirect(url_for("dashboard"))

@app.route("/delete_all", methods=["GET", "POST"])
def delete_all():
    if not session.get("user_email"): return redirect(url_for("login"))
    for scan in list(scans_col.find({"user_email": session["user_email"]})):
        try: os.remove(os.path.join(app.config["UPLOAD_FOLDER"], scan.get("filename", "")))
        except: pass
        try: os.remove(os.path.join(app.config["UPLOAD_FOLDER"], scan.get("heatmap_filename", "")))
        except: pass
    scans_col.delete_many({"user_email": session["user_email"]})
    return redirect(url_for("dashboard"))

@app.route("/delete/<scan_id>", methods=["GET", "POST"])
def delete_scan(scan_id):
    if not session.get("user_email"): return redirect(url_for("login"))
    scan = scans_col.find_one({"_id": ObjectId(scan_id)})
    if scan:
        try: os.remove(os.path.join(app.config["UPLOAD_FOLDER"], scan.get("filename", "")))
        except: pass
        try: os.remove(os.path.join(app.config["UPLOAD_FOLDER"], scan.get("heatmap_filename", "")))
        except: pass
        scans_col.delete_one({"_id": ObjectId(scan_id)})
    return redirect(url_for("dashboard"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("user_email"): return redirect(url_for("login"))
    purge_old_uploads(app.config["UPLOAD_FOLDER"])

    if request.method == "POST":
        file = request.files.get('image')
        
        if not file or file.filename == '':
            flash("No image selected.", "error")
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            # --- FRONTEND FIX: Save original PRISTINE image to hard drive. 
            # We no longer overwrite this with a compressed/blurry version.
            file.save(original_path)

            display_filename = filename
            analysis_path = original_path

            if filename.lower().endswith('.pdf'):
                extracted_name, extracted_path = handle_pdf(original_path, filename)
                if extracted_name:
                    display_filename = extracted_name
                    analysis_path = extracted_path
                else:
                    flash("Failed to extract image from PDF.", "error")
                    return redirect(request.url)

            # CORE AI SYSTEM EXECUTES HERE
            label, prob, ela, noise, exif, freq, edge, heatmap_name, raw_explanation = forensic_predict(analysis_path)
            
            # --- STRICT LABEL ENFORCER ---
            # Guarantees the frontend never breaks if Gemini hallucinates the category name
            label_upper = label.upper()
            if "TEXT" in label_upper or "MANIPULATED" in label_upper or "EDITED" in label_upper:
                label = "Manipulated / Edited"
            elif "AI" in label_upper or "GENERATED" in label_upper:
                label = "AI Generated"
            else:
                label = "Authentic"

            # --- CRASH FIX FOR GEMINI LIST HALLUCINATIONS ---
            if isinstance(raw_explanation, list):
                raw_explanation = "\n".join(str(item) for item in raw_explanation)
            elif not isinstance(raw_explanation, str):
                raw_explanation = str(raw_explanation)
            
            explanation_html = markdown.markdown(raw_explanation)

            scans_col.insert_one({
                "user_email": session["user_email"], 
                "filename": display_filename, 
                "heatmap_filename": heatmap_name, 
                "prediction": label, 
                "confidence": prob, 
                "ela_score": ela, 
                "noise_variance": noise,
                "freq_score": freq, 
                "edge_score": edge, 
                "exif_status": exif, 
                "explanation": explanation_html, 
                "timestamp": datetime.now(timezone.utc)
            })

            return render_template("result.html", label=label, prob=prob, ela=ela, noise=noise, exif=exif,
                                   freq=freq, edge=edge, explanation=explanation_html, image=display_filename, heatmap=heatmap_name)
        else:
            flash("Invalid file type uploaded.", "error")
            return redirect(request.url)
                                   
    return render_template("upload.html")

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("File too large. Please upload an image smaller than 16MB.", "error")
    return redirect(url_for('upload')), 413

if __name__ == "__main__":
    app.run(debug=True)