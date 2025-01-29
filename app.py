import os
import joblib
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import pandas as pd
import random
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte
import cv2
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
app.secret_key = "supersecretkey"

# إعداد مجلدات التخزين
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # تحديد الحد الأقصى لحجم الملفات بـ 16MB

# تحديد مسار الملف
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fruit_classifier_improved_5features = os.path.join(BASE_DIR, "models", "fruit_classifier_improved_5features.pkl")
MODEL_CNN_PATH = os.path.join(BASE_DIR, "models", "fruit_classifier_cnn.h5")

# التحقق من وجود الملف قبل تحميله
if os.path.exists(fruit_classifier_improved_5features):
    rf_model = joblib.load(fruit_classifier_improved_5features)  # تحميل باستخدام joblib
else:
    rf_model = None

# تحميل النموذج CNN
cnn_model = load_model(MODEL_CNN_PATH) if os.path.exists(MODEL_CNN_PATH) else None

print("fruit_classifier_improved_5features:", fruit_classifier_improved_5features)
print("File exists:", os.path.exists(fruit_classifier_improved_5features))

# قائمة الفواكه التي يعرفها الموديل
FRUITS = ["Apple", "Banana", "Orange"]

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(img_as_ubyte(gray), [1], [0], symmetric=True, normed=True)
    
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    
    return contrast, correlation, energy

def classify_fruit(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return 'No Fruit Detected', 'Unknown', 'Unknown', 'Unknown'

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 'No Fruit Detected', 'Unknown', 'Unknown', 'Unknown'

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    white_pixel_ratio = white_pixels / total_pixels

    edges = cv2.Canny(gray_image, 80, 200)
    edge_pixels = np.sum(edges == 255)
    edge_pixel_ratio = edge_pixels / total_pixels

    contrast, correlation, energy = extract_texture_features(image)
    mean_intensity = np.mean(gray_image)

    features = np.array([[white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy, mean_intensity]])

    # استخدام CNN إذا كان متاحًا
    if cnn_model:
        cnn_predictions = cnn_model.predict(features)
        predicted_quality_cnn = 'fresh' if cnn_predictions[0][0] < 0.5 else 'rotten'
        predicted_fruit_cnn = FRUITS[np.argmax(cnn_predictions[0][1:])]
    else:
        predicted_quality_cnn = 'Unknown'
        predicted_fruit_cnn = 'Unknown'

    # استخدام RandomForest إذا كان متاحًا
    if rf_model:
        rf_predictions = rf_model.predict(features)
        predicted_class_rf = rf_predictions[0]
        predicted_fruit_rf = FRUITS[predicted_class_rf // 2]
        predicted_quality_rf = 'fresh' if predicted_class_rf % 2 == 0 else 'rotten'
    else:
        predicted_quality_rf = 'Unknown'
        predicted_fruit_rf = 'Unknown'

    return predicted_fruit_cnn, predicted_quality_cnn, predicted_fruit_rf, predicted_quality_rf

def process_and_save_image(image, filename):
    max_size = (800, 800)
    image = image.convert("RGB")
    image.thumbnail(max_size, Image.LANCZOS)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(save_path, "JPEG", quality=85)
    return filename

@app.route('/analyze', methods=['POST'])
def analyze():
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
        
        # تحليل الصورة باستخدام النموذج
        fruit_type_cnn, quality_cnn, fruit_type_rf, quality_rf = classify_fruit(file_path)

        return render_template('result.html', 
                       fruit_type_rf=fruit_type_rf, 
                       quality_rf=quality_rf,
                       image_url=url_for('uploaded_file', filename=filename), 
                       random_value=random.randint(0, 10000))

@app.route('/analyze_camera', methods=['POST'])
def analyze_camera():
    image_data = request.form['image_data'].replace("data:image/jpeg;base64,", "")
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # حفظ الصورة في ملف
    filename = f"captured_{random.randint(1000, 9999)}.jpg"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(file_path)
    
    # تحليل الصورة باستخدام النموذج
    fruit_type_cnn, quality_cnn, fruit_type_rf, quality_rf = classify_fruit(file_path)
    
    return render_template('result.html', 
                           fruit_type_rf=fruit_type_rf, 
                           quality_rf=quality_rf, 
                           image_url=url_for('uploaded_file', filename=filename), 
                           random_value=random.randint(0, 10000))

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in request"}), 400
    
    image_file = request.files['image']
    processed_image = preprocess_image(image_file)  # تحويل الصورة لتنسيق صالح للنموذج
    result = model.predict(processed_image)
    
    # إرجاع النتيجة كـ JSON
    return jsonify({"prediction": result.tolist()})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dashboard')
def dashboard():
    feedback_file = os.path.join(RESULTS_FOLDER, "user_feedback.csv")
    if not os.path.exists(feedback_file):
        return render_template("dashboard.html", total_images=0, fresh_percentage=0, medium_percentage=0, rotten_percentage=0, user_feedback_count=0)

    try:
        feedback_data = pd.read_csv(feedback_file)
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return render_template("dashboard.html", total_images=0, fresh_percentage=0, medium_percentage=0, rotten_percentage=0, user_feedback_count=0)

    required_columns = ["filename", "fruit_type", "quality", "user_feedback", "correct_fruit", "correct_quality"]
    if not all(column in feedback_data.columns for column in required_columns):
        print("CSV file does not contain the required columns")
        return render_template("dashboard.html", total_images=0, fresh_percentage=0, medium_percentage=0, rotten_percentage=0, user_feedback_count=0)

    total_images = len(feedback_data)
    fresh_percentage = (feedback_data['correct_quality'].value_counts().get("fresh", 0) / total_images) * 100 if total_images else 0
    medium_percentage = (feedback_data['correct_quality'].value_counts().get("medium", 0) / total_images) * 100 if total_images else 0
    rotten_percentage = (feedback_data['correct_quality'].value_counts().get("rotten", 0) / total_images) * 100 if total_images else 0
    user_feedback_count = feedback_data[feedback_data['user_feedback'] == 'incorrect'].shape[0] if total_images else 0

    return render_template("dashboard.html", total_images=total_images, fresh_percentage=round(fresh_percentage, 2), medium_percentage=round(medium_percentage, 2), rotten_percentage=round(rotten_percentage, 2), user_feedback_count=user_feedback_count)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    filename = request.form['filename']
    fruit_type = request.form['fruit_type']
    quality = request.form['quality']
    user_feedback = request.form['user_feedback']
    correct_fruit = request.form.get('correct_fruit', None)
    correct_quality = request.form.get('correct_quality', None)

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
    
    # التحقق من وجود الملف وإضافة الأعمدة إذا لم يكن موجودًا
    if not os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode='a', header=True, index=False)
    else:
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)

    flash('Thank you for your feedback!')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
