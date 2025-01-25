import cv2
import numpy as np
import os
import pandas as pd

# الحصول على المسار الأساسي للمشروع
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# تحديد مسارات مجموعة البيانات
dataset_path = os.path.join(base_dir, "dataset", "train")
output_csv = os.path.join(base_dir, "results", "features.csv")

# التأكد من وجود مجلد النتائج
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

# تهيئة قائمة لتخزين البيانات
data = []

# قراءة جميع الصور من الفئات المختلفة
categories = ["apple", "banana", "orange"]
labels = ["fresh", "rotten"]

for category in categories:
    for label in labels:
        folder_path = os.path.join(dataset_path, category, label)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ المجلد غير موجود: {folder_path}")
            continue
        
        # قراءة كل الصور في المجلد
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            # تحميل الصورة
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ لم يتم تحميل الصورة: {image_path}")
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

            # إضافة الميزات إلى القائمة
            data.append([category, label, filename, white_pixel_ratio, edge_pixel_ratio])

# تحويل البيانات إلى DataFrame
df = pd.DataFrame(data, columns=["Category", "Label", "Filename", "WhitePixelRatio", "EdgePixelRatio"])

# حفظ البيانات في ملف CSV
df.to_csv(output_csv, index=False)
print(f"✅ تم استخراج الميزات وحفظها في: {output_csv}")
