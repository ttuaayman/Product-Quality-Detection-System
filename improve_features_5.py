import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import greycomatrix, greycoprops  # أو graycomatrix, graycoprops
from skimage import img_as_ubyte

# تعريف دالة استخراج ميزات النسيج
def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(img_as_ubyte(gray), [1], [0], symmetric=True, normed=True)
    
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    
    return contrast, correlation, energy

# المسار الأساسي للمشروع
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "dataset", "train")
output_csv = os.path.join(base_dir, "results", "features_5improved.csv")

# التأكد من وجود مجلد النتائج
results_path = os.path.join(base_dir, "results")
os.makedirs(results_path, exist_ok=True)

# الفئات والتصنيفات
categories = ["apple", "banana", "orange"]
labels = ["fresh", "rotten"]
data = []

# استخراج الميزات من كل صورة
for category in categories:
    for label in labels:
        folder_path = os.path.join(dataset_path, category, label)

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            # تحويل الصورة إلى تدرج الرمادي
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # تطبيق Otsu's Thresholding
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_pixels = np.sum(thresh == 255)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            white_pixel_ratio = white_pixels / total_pixels

            # تطبيق Canny Edge Detection
            edges = cv2.Canny(gray_image, 80, 200)
            edge_pixels = np.sum(edges == 255)
            edge_pixel_ratio = edge_pixels / total_pixels

            # استخراج ميزات النسيج
            contrast, correlation, energy = extract_texture_features(image)

            # حساب متوسط شدة اللون
            mean_intensity = np.mean(gray_image)

            # إضافة البيانات إلى القائمة
            data.append([category, label, filename, white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy, mean_intensity])

# تحويل البيانات إلى DataFrame
df_features = pd.DataFrame(data, columns=["Category", "Label", "Filename", "WhitePixelRatio", "EdgePixelRatio", "Contrast", "Correlation", "Energy", "MeanIntensity"])

# حفظ البيانات في ملف CSV
df_features.to_csv(output_csv, index=False)
print(f"\n✅ 6 Features saved to: {output_csv}")