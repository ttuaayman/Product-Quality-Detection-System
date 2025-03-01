import cv2
import numpy as np
import os
import pandas as pd
import joblib

# تحميل النموذج المحسن
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "fruit_classifier_improved_5features.pkl")

if not os.path.exists(model_path):
    print("❌ Model file not found!")
    exit()

model = joblib.load(model_path)

# تحديد مسار بيانات الاختبار
test_images_path = os.path.join(base_dir, "dataset", "test")

# قراءة جميع الصور من مجلد الاختبار
categories = ["apple", "banana", "orange"]
labels = ["fresh", "rotten"]
data = []

for category in categories:
    for label in labels:
        folder_path = os.path.join(test_images_path, category, label)

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"❌ Error loading image: {image_path}")
                continue

            # استخراج الميزات الجديدة
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Otsu's Thresholding
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_pixels = np.sum(thresh == 255)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            white_pixel_ratio = white_pixels / total_pixels

            # Canny Edge Detection
            edges = cv2.Canny(gray_image, 80, 200)
            edge_pixels = np.sum(edges == 255)
            edge_pixel_ratio = edge_pixels / total_pixels

            # Haralick Texture Features
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

            # تنبؤ الفئة باستخدام النموذج المحسن
            prediction = model.predict([[white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy, homogeneity]])[0]
            predicted_label = "fresh" if prediction == 0 else "rotten"

            data.append([category, label, filename, predicted_label])

# تحويل النتائج إلى DataFrame وعرضها
df_results = pd.DataFrame(data, columns=["Category", "True Label", "Filename", "Predicted Label"])
print("\n🔍 **Test Results:**")
print(df_results.head())

# حساب دقة النموذج على بيانات الاختبار
accuracy = (df_results["True Label"] == df_results["Predicted Label"]).mean()
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# حفظ النتائج في ملف CSV
results_csv_path = os.path.join(base_dir, "results", "test_results_improved.csv")
df_results.to_csv(results_csv_path, index=False)
print(f"\n✅ Results saved to: {results_csv_path}")