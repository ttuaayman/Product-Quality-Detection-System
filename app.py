import os
import base64
import random
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename

# Keras imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ====== Paths and Directories ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ====== Model Path (CNN) ======
# If you have a different filename, change "fruit_mobilenetv2.keras" to your actual model file
MODEL_CNN_PATH = os.path.join(BASE_DIR, "models", "fruit_mobilenetv2.keras")

# ====== Load the CNN model ======
if os.path.exists(MODEL_CNN_PATH):
    cnn_model = load_model(MODEL_CNN_PATH)
    print(f"✅ Loaded CNN model from: {MODEL_CNN_PATH}")
else:
    cnn_model = None
    print(f"❌ Model file not found at: {MODEL_CNN_PATH}")

# ====== Class Names ======
# Suppose you have 6 classes: apple_fresh, apple_rotten, banana_fresh, banana_rotten, orange_fresh, orange_rotten
class_names = [
    "apple_fresh",
    "apple_rotten",
    "banana_fresh",
    "banana_rotten",
    "orange_fresh",
    "orange_rotten"
]

def parse_class_name(class_str):
    """
    Example: 'apple_fresh' -> ('apple', 'fresh').
    If format is unexpected, returns ('Unknown', 'Unknown').
    """
    parts = class_str.split("_")
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return "Unknown", "Unknown"

def preprocess_image_for_cnn(image_path, target_size=(224, 224)):
    """
    Loads an image from disk, resizes to target_size, normalizes to [0..1],
    and expands dimensions to [1, height, width, channels].
    Adjust target_size if your model was trained on a different size (e.g. (150,150)).
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def classify_fruit(image_path):
    """
    Classifies a single image using the loaded CNN model.
    Returns (fruit_name, quality_name).
    If confidence < THRESHOLD, returns ("No Fruit", "Detected").
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return ("Unknown", "Unknown")

    if cnn_model is None:
        print("❌ CNN model is not loaded.")
        return ("Unknown", "Unknown")

    # 1) Preprocess
    processed_img = preprocess_image_for_cnn(image_path, target_size=(224, 224))

    # 2) Predict
    predictions = cnn_model.predict(processed_img)  # shape: (1, 6)
    predicted_idx = np.argmax(predictions[0])       # index of the highest probability
    confidence = predictions[0][predicted_idx]      # the probability

    # 3) Threshold to handle empty images
    THRESHOLD = 0.7  # you can tweak this value
    if confidence < THRESHOLD:
        # Treat it as no fruit
        return ("No Fruit", "Detected")

    # 4) Map the class index to (fruit, quality)
    class_str = class_names[predicted_idx]
    fruit_name, quality_name = parse_class_name(class_str)
    return (fruit_name, quality_name)

# ====================== Flask Routes ====================== #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Receives an uploaded image, saves it, classifies with CNN, and shows the result.
    """
    if 'file' not in request.files:
        flash("No file part in request")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Classify
        fruit_name, quality_name = classify_fruit(file_path)

        # Render the result page
        return render_template(
            'result.html',
            fruit_type=fruit_name,
            quality=quality_name,
            filename=filename,
            image_url=url_for('uploaded_file', filename=filename),
            random_value=random.randint(0, 10000)
        )

@app.route('/analyze_camera', methods=['POST'])
def analyze_camera():
    """
    Similar to /analyze but receives a base64 image from camera.
    """
    image_data = request.form['image_data'].replace("data:image/jpeg;base64,", "")
    image_bytes = base64.b64decode(image_data)
    pil_img = Image.open(BytesIO(image_bytes))

    filename = f"captured_{random.randint(1000, 9999)}.jpg"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pil_img.save(file_path)

    fruit_name, quality_name = classify_fruit(file_path)

    return render_template(
        'result.html',
        fruit_type=fruit_name,
        quality=quality_name,
        filename=filename,
        image_url=url_for('uploaded_file', filename=filename),
        random_value=random.randint(0, 10000)
    )

@app.route('/dashboard')
def dashboard():
    """
    Displays stats from user_feedback.csv, including 'No Fruit Detected' if you want to track it.
    """
    feedback_file = os.path.join(RESULTS_FOLDER, "user_feedback.csv")
    if not os.path.exists(feedback_file):
        # No file => return zero stats
        return render_template(
            "dashboard.html",
            total_images=0,
            fresh_percentage=0,
            medium_percentage=0,
            rotten_percentage=0,
            # If you want to show no_fruit_percentage
            no_fruit_percentage=0,
            user_feedback_count=0
        )

    try:
        feedback_data = pd.read_csv(feedback_file)
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return render_template(
            "dashboard.html",
            total_images=0,
            fresh_percentage=0,
            medium_percentage=0,
            rotten_percentage=0,
            no_fruit_percentage=0,
            user_feedback_count=0
        )

    required_columns = ["filename", "fruit_type", "quality", "user_feedback", "correct_fruit", "correct_quality"]
    if not all(column in feedback_data.columns for column in required_columns):
        print("CSV file does not contain the required columns")
        return render_template(
            "dashboard.html",
            total_images=0,
            fresh_percentage=0,
            medium_percentage=0,
            rotten_percentage=0,
            no_fruit_percentage=0,
            user_feedback_count=0
        )

    total_images = len(feedback_data)
    if total_images == 0:
        return render_template(
            "dashboard.html",
            total_images=0,
            fresh_percentage=0,
            medium_percentage=0,
            rotten_percentage=0,
            no_fruit_percentage=0,
            user_feedback_count=0
        )

    # Count each category in 'correct_quality'
    fresh_count = feedback_data['correct_quality'].value_counts().get("fresh", 0)
    medium_count = feedback_data['correct_quality'].value_counts().get("medium", 0)
    rotten_count = feedback_data['correct_quality'].value_counts().get("rotten", 0)
    no_fruit_count = feedback_data['correct_quality'].value_counts().get("No Fruit Detected", 0)

    fresh_percentage = (fresh_count / total_images) * 100
    medium_percentage = (medium_count / total_images) * 100
    rotten_percentage = (rotten_count / total_images) * 100
    no_fruit_percentage = (no_fruit_count / total_images) * 100

    user_feedback_count = feedback_data[feedback_data['user_feedback'] == 'incorrect'].shape[0]

    return render_template(
        "dashboard.html",
        total_images=total_images,
        fresh_percentage=round(fresh_percentage, 2),
        medium_percentage=round(medium_percentage, 2),
        rotten_percentage=round(rotten_percentage, 2),
        no_fruit_percentage=round(no_fruit_percentage, 2),
        user_feedback_count=user_feedback_count
    )

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Receives user feedback from result.html and stores it in user_feedback.csv.
    """
    filename = request.form['filename']
    fruit_type = request.form['fruit_type']
    quality = request.form['quality']
    user_feedback = request.form['user_feedback']
    correct_fruit = request.form.get('correct_fruit', None)
    correct_quality = request.form.get('correct_quality', None)

    # Create a record to store
    feedback_data = {
        'filename': filename,
        'fruit_type': fruit_type,
        'quality': quality,
        'user_feedback': user_feedback,
        'correct_fruit': correct_fruit,
        'correct_quality': correct_quality
    }

    feedback_df = pd.DataFrame([feedback_data])
    feedback_file = os.path.join(RESULTS_FOLDER, "user_feedback.csv")

    if not os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode='a', header=True, index=False)
    else:
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)

    flash('Thank you for your feedback!')
    return redirect(url_for('dashboard'))

# Additional info pages
@app.route('/breda_robotics')
def breda_robotics():
    return render_template("info.html", title="Breda Robotics", content="Breda Robotics specializes in AI-driven automation solutions for industrial applications.")

@app.route('/utrecht_university')
def utrecht_university():
    return render_template("info.html", title="Utrecht University", content="Utrecht University is a leading institution in applied sciences, fostering innovation in engineering and AI research.")

@app.route('/next_level_engineering')
def next_level_engineering():
    return render_template("info.html", title="Next Level Engineering", content="This Master's program focuses on AI, robotics, and automation to develop future-ready solutions.")

@app.route('/project_info')
def project_info():
    return render_template("info.html", title="Project Overview", content="This project leverages AI and computer vision to automate quality control in manufacturing and retail.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
