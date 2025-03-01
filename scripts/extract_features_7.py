import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte

# تحديد المسارات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# اسم ملف CSV الناتج
FEATURES_CSV = os.path.join(RESULTS_DIR, "features_7combined.csv")

# قائمة الفواكه
FRUITS = ["apple", "banana", "orange"]

def extract_features(image_path):
    """
    ترجع دالة استخراج الميزات قائمة تحتوي على:
    [white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy, mean_intensity, variance]
    أو None في حال فشل قراءة الصورة.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ فشل في قراءة الصورة: {image_path}")
        return None
    
    # تحويل الصورة إلى تدرج الرمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) Otsu's Thresholding لحساب white_pixel_ratio
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    white_pixel_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

    # 2) Canny Edge Detection لحساب edge_pixel_ratio
    edges = cv2.Canny(gray, 80, 200)
    edge_pixels = np.sum(edges == 255)
    edge_pixel_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0

    # 3) GLCM لحساب Contrast, Correlation, Energy
    glcm = greycomatrix(img_as_ubyte(gray), [1], [0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]

    # 4) Mean Intensity
    mean_intensity = np.mean(gray) / 255.0  # نقسم على 255 لتطبيع القيم إلى [0,1]

    # 5) Variance
    variance = np.var(gray) / 255.0  # يمكن إبقاءه دون قسمة أيضاً

    return [
        white_pixel_ratio,
        edge_pixel_ratio,
        contrast,
        correlation,
        energy,
        mean_intensity,
        variance
    ]

# استخراج الميزات لجميع الصور في مجلد train
features_data = []

for fruit in FRUITS:
    for quality in ["fresh", "rotten"]:
        quality_dir = os.path.join(DATASET_DIR, fruit, quality)
        if not os.path.exists(quality_dir):
            print(f"⚠️ المجلد غير موجود: {quality_dir}")
            continue
        
        for image_name in os.listdir(quality_dir):
            image_path = os.path.join(quality_dir, image_name)
            
            # استخرج الميزات
            feats = extract_features(image_path)
            if feats is not None:
                # feats عبارة عن قائمة من 7 قيم
                white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy, mean_intensity, variance = feats
                
                # خزّنها في قائمة
                features_data.append([
                    fruit,
                    quality,
                    image_name,
                    white_pixel_ratio,
                    edge_pixel_ratio,
                    contrast,
                    correlation,
                    energy,
                    mean_intensity,
                    variance
                ])

# تحويل الميزات إلى DataFrame
columns = [
    "FruitType",
    "Label",
    "Filename",
    "WhitePixelRatio",
    "EdgePixelRatio",
    "Contrast",
    "Correlation",
    "Energy",
    "MeanIntensity",
    "Variance"
]
df = pd.DataFrame(features_data, columns=columns)

# حفظ الميزات في ملف CSV
df.to_csv(FEATURES_CSV, index=False)
print(f"✅ تم استخراج الميزات وحفظها في: {FEATURES_CSV}")

