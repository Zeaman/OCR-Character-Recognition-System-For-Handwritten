# OCR-Character-Recognition-System-For-Handwritten
A Python-based application that recognizes handwritten alphanumeric characters (A-Z, 0-9) using deep learning. The system allows users to draw characters or upload images for prediction.

## 1.  Features

- ✍️ **Interactive Drawing Canvas** with adjustable brush size
- 📁 **Image Upload** (PNG/JPG/JPEG)
- 🔍 **ResNet18-based Classifier** trained on custom dataset
- 📊 **Top-5 Predictions** with confidence scores
- 🖥️ **CPU-Only Execution** for broad compatibility
- 📱 **Responsive Tkinter GUI**

## Installation

### 1. Clone the repository:

git clone https://github.com/Zeaman/OCR-Character-Recognition-System-For-Handwritten.git
cd OCR-Character-Recognition-System-For-Handwritten

### 2. Create and activate a virtual environment (Windows):

python -m venv Myvenv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

## For Regular Usage

# Activate environment (when returning to project)
cd OCR-Character-Recognition-System-For-Handwritten
venv\Scripts\activate

# Run the application
python src/ocr_app.py

##  Common Git Commands

# After making changes
git add .
git commit -m "Your commit message"
git push origin main

# To pull the latest changes
git pull origin main

# Expected GUI (TKinter GUI):


