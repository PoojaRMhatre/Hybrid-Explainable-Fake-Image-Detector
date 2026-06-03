#Digital Forensics & Deepfake Detection Platform

Welcome to our project, Aletheia—an advanced, hybrid digital forensics web application designed to detect manipulated images, deepfakes, and AI-generated content. 

By combining cutting-edge Deep Learning (PyTorch) with traditional digital forensic algorithms and Large Language Models (Gemini), our project provides highly accurate, explainable results to users through a secure and responsive web dashboard.

## 🚀 Key Features

* **Deep Learning Classification**: Utilizes a custom-trained PyTorch Convolutional Neural Network (based on MobileNetV2) to classify images as Authentic, AI-Generated, or Manipulated.
* **Error Level Analysis (ELA) & Thermal Heatmaps**: Analyzes JPEG compression rates and generates visual thermal heatmaps using OpenCV to highlight spliced or edited regions.
* **Advanced Noise Variance Estimation**: Employs Immerkær’s Fast Noise Variance algorithm to detect unnatural pixel smoothness commonly found in AI-generated images (e.g., Midjourney, Stable Diffusion).
* **EXIF Metadata Forensics**: Scrapes and analyzes image metadata to verify authentic camera hardware signatures and flag known editing software traces.
* **Explainable AI (XAI) Reports**: Integrates Google's Gemini 2.5 Flash model to synthesize the forensic metrics into a concise, 3-bullet point technical explanation for the user.
* **Secure User Authentication**: Features a robust login system with encrypted passwords, time-sensitive email verification, and secure password reset links.
* **Personalized Analytics Dashboard**: Logged-in users get a dedicated dashboard saving their scan history, complete with statistical metrics and the ability to re-view past forensic reports.
* **PDF Extraction Support**: Automatically extracts and analyzes the first image found within uploaded PDF documents.

## 🛠️ Tech Stack

* **Backend Framework**: Python, Flask
* **Machine Learning**: PyTorch, Torchvision
* **Computer Vision & Image Processing**: OpenCV, Pillow (PIL), NumPy
* **Database**: MongoDB
* **Generative AI**: Google Generative AI (Gemini 2.5 Flash API)
* **Frontend**: HTML5, CSS3, Jinja2 (with Glassmorphism UI elements)
* **Security & Auth**: Werkzeug Security, Flask-Mail, Itsdangerous (Encrypted Tokens)

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your system:
* Python 3.8+
* MongoDB (Running locally on port `27017` or via MongoDB Atlas)
* A Google account (for Gemini API and Gmail App Passwords)

### 2. Clone the Repository
```bash
git clone [https://github.com/yourusername/aletheia-forensics.git](https://github.com/yourusername/aletheia-forensics.git)
cd aletheia-forensics