from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import pandas as pd
import random
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from skimage.feature import greycomatrix, greycoprops  # Adding feature extraction libraries
from skimage import img_as_ubyte
import cv2
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, classification_report

from scripts.improve_features import extract_texture_features  # Import OpenCV

print("Current Working Directory:", os.getcwd())

app = Flask(__name__)
app.secret_key = "supersecretkey"

# 📂 إعداد مجلدات التخزين
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # تحديد الحد الأقصى لحجم الملفات بـ 16MB

# ✅ تحميل الموديل المدرب
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fruit_classifier.h5")
model = load_model(MODEL_PATH)

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
        return 'No Fruit Detected', 'Unknown'

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 'No Fruit Detected', 'Unknown'

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

    # تحميل النموذج
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fruit_classifier.h5")
    model = load_model(model_path)

    # التنبؤ
    predictions = model.predict(features)
    predicted_quality = 'fresh' if predictions[0][0] < 0.5 else 'rotten'
    
    # التحقق من وجود تنبؤات لأنواع الفاكهة
    if len(predictions[0]) > 1:
        predicted_fruit_type = np.argmax(predictions[0][1:])  # افترض أن الأنواع المختلفة من الفواكه تبدأ من الفهرس 1
        fruit_types = ['Apple', 'Banana', 'Orange', 'Strawberry', 'Mango', 'Grapes']  # قائمة بأنواع الفواكه
        predicted_fruit_type = fruit_types[predicted_fruit_type] if predicted_fruit_type < len(fruit_types) else 'Unknown'
    else:
        predicted_fruit_type = 'No Fruit Detected'

    return predicted_fruit_type, predicted_quality

def process_and_save_image(image, filename):
    """ تصغير الصورة مع الحفاظ على جودتها """
    max_size = (800, 800)
    image = image.convert("RGB")
    image.thumbnail(max_size, Image.LANCZOS)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(save_path, "JPEG", quality=85)
    return filename

# 🟢 تحميل النموذج
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "fruit_classifier.h5")
model = load_model(model_path)

# 📸 تحليل الصورة من الجهاز
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        fruit_type, quality = classify_fruit(file_path)
        return render_template('result.html', fruit_type=fruit_type, quality=quality, image_url=url_for('uploaded_file', filename=filename), random_value=random.randint(0, 10000))

# 📷 تحليل الصورة من الكاميرا
@app.route('/analyze_camera', methods=['POST'])
def analyze_camera():
    image_data = request.form['image_data'].replace("data:image/jpeg;base64,", "")

    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))

    filename = f"captured_{random.randint(1000, 9999)}.jpg"
    filename = process_and_save_image(image, filename)
    fruit_type, quality = classify_fruit(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('result.html', fruit_type=fruit_type, quality=quality)

# 📂 عرض الصور المحفوظة
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 📊 لوحة التحكم
@app.route('/dashboard')
def dashboard():
    feedback_file = os.path.join(RESULTS_FOLDER, "user_feedback.csv")

    if not os.path.exists(feedback_file):
        return render_template("dashboard.html", total_images=0, fresh_percentage=0, medium_percentage=0, rotten_percentage=0, user_feedback_count=0)

    feedback_data = pd.read_csv(feedback_file, names=["filename", "predicted", "feedback", "correct"])

    total_images = len(feedback_data)
    fresh_percentage = (feedback_data['correct'].value_counts().get("Fresh", 0) / total_images) * 100 if total_images else 0
    medium_percentage = (feedback_data['correct'].value_counts().get("Medium", 0) / total_images)  * 100 if total_images else 0
    rotten_percentage = (feedback_data['correct'].value_counts().get("Rotten", 0) / total_images) * 100 if total_images else 0
    user_feedback_count = feedback_data[feedback_data['feedback'] == 'incorrect'].shape[0] if total_images else 0

    return render_template("dashboard.html",
                           total_images=total_images,
                           fresh_percentage=round(fresh_percentage, 2),
                           medium_percentage=round(medium_percentage, 2),
                           rotten_percentage=round(rotten_percentage, 2),
                           user_feedback_count=user_feedback_count)

# 🏢 معلومات عن الشركة
@app.route('/breda_robotics')
def breda_robotics():
    return render_template("info.html", title="Breda Robotics", content="Breda Robotics specializes in AI-driven automation solutions for industrial applications.")

# 🎓 معلومات عن الجامعة
@app.route('/utrecht_university')
def utrecht_university():
    return render_template("info.html", title="Utrecht University", content="Utrecht University is a leading institution in applied sciences, fostering innovation in engineering and AI research.")

# 📚 معلومات عن الماستر
@app.route('/next_level_engineering')
def next_level_engineering():
    return render_template("info.html", title="Next Level Engineering", content="This Master's program focuses on AI, robotics, and automation to develop future-ready solutions.")

# 💡 معلومات عن المشروع
@app.route('/project_info')
def project_info():
    return render_template("info.html", title="Project Overview", content="This project leverages AI and computer vision to automate quality control in manufacturing and retail.")

# 🏠 الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')  # تأكد من أن لديك هذا الملف في مجلد templates

# 🚀 تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)

# دالة لتحميل النموذج
def load_trained_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fruit_classifier.h5")
    model = load_model(model_path)
    return model

# دالة لاستخراج الميزات من الصورة
def extract_features(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

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

    # استخراج ميزات الألوان
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))

    features = np.array([white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy, mean_intensity, *mean_color, *std_color])
    return features

# دالة لاختبار الدقة
def test_model_accuracy(model, test_data):
    X_test = []
    y_test = []

    for image_path, label in test_data:
        features = extract_features(image_path)
        if features is not None:
            X_test.append(features)
            y_test.append(label)

    if not X_test:
        print("No valid test data found.")
        return None, None

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    predictions = model.predict(X_test)
    predicted_labels = ['fresh' if pred < 0.5 else 'rotten' for pred in predictions]

    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

    return accuracy, report

# تحميل النموذج
model = load_trained_model()

# تحميل مجموعة البيانات (تأكد من أن لديك مجموعة بيانات اختبار)
test_data = [
    ("path/to/test_image1.jpg", "fresh"),
    ("path/to/test_image2.jpg", "rotten"),
    # أضف المزيد من الصور والملصقات هنا
]

# اختبار الدقة
accuracy, report = test_model_accuracy(model, test_data)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# إعداد البيانات
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'path/to/dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    'path/to/dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# بناء نموذج CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب النموذج
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# حفظ النموذج
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fruit_classifier_cnn.h5")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

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
    feedback_df.to_csv(feedback_file, mode='a', header=not os.path.exists(feedback_file), index=False)

    flash('Thank you for your feedback!')
    return redirect(url_for('dashboard'))
