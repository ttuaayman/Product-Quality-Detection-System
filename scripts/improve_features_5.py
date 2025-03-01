import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import greycomatrix, greycoprops  # أو graycomatrix, graycoprops
from skimage import img_as_ubyte

# تحديد المسارات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FEATURES_CSV = os.path.join(RESULTS_DIR, "features_5improved.csv")

# قائمة الفواكه
FRUITS = ["apple", "banana", "orange"]

# دالة لاستخراج الميزات
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(img_as_ubyte(gray), [1], [0], symmetric=True, normed=True)
    
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    
    return contrast, correlation, energy

# استخراج الميزات لجميع الصور
features = []
for fruit in FRUITS:
    for quality in ["fresh", "rotten"]:
        quality_dir = os.path.join(DATASET_DIR, fruit, quality)
        if not os.path.exists(quality_dir):
            print(f"Directory {quality_dir} does not exist")
            continue
        for image_name in os.listdir(quality_dir):
            image_path = os.path.join(quality_dir, image_name)
            contrast, correlation, energy = extract_features(image_path)
            if contrast is not None and correlation is not None and energy is not None:
                features.append([fruit, quality, contrast, correlation, energy])

# تحويل الميزات إلى DataFrame
df = pd.DataFrame(features, columns=["FruitType", "Label", "Contrast", "Correlation", "Energy"])

# حفظ الميزات في ملف CSV
df.to_csv(FEATURES_CSV, index=False)
print(f"Features saved to {FEATURES_CSV}")
