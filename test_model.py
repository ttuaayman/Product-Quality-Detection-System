import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model # type: ignore
import cv2

from scripts.improve_features import extract_texture_features

# دالة لتحميل النموذج
def load_trained_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fruit_classifier.h5")
    model = load_model(model_path)
    return model

# دالة لاستخراج الميزات من الصورة
def extract_features(image_path):
    image = cv2.imread(image_path)
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

    features = np.array([white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy, mean_intensity])
    return features

# دالة لاختبار الدقة
def test_model_accuracy(model, test_data):
    X_test = []
    y_test = []

    for image_path, label in test_data:
        features = extract_features(image_path)
        X_test.append(features)
        y_test.append(label)

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