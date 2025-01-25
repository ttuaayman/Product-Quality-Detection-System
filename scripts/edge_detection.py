import cv2
import numpy as np
import os

# الحصول على المسار الأساسي للمشروع
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# تحديد المسار الصحيح للصورة
image_path = os.path.join(base_dir, "dataset", "sample.jpg")

# التحقق مما إذا كان الملف موجودًا
if not os.path.exists(image_path):
    print(f"❌ لم يتم العثور على الصورة في المسار: {image_path}")
    print("🔹 تأكد من أن الصورة موجودة داخل مجلد 'dataset/' واسمها 'sample.jpg'.")
    exit()

# تحميل الصورة
image = cv2.imread(image_path)

# تحويل الصورة إلى تدرج الرمادي
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# تطبيق مرشح التمويه لتقليل الضوضاء
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

### 🚀 **تحسين اكتشاف الحواف باستخدام Canny** ###
# ضبط القيم الجديدة للحواف لتقليل الضوضاء وزيادة وضوح الشكل
edges = cv2.Canny(blurred_image, 80, 200)  # بدلاً من (50, 150)

### 🚀 **تطبيق Otsu's Thresholding كبديل عن Canny** ###
_, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# التأكد من وجود مجلد "results/"
results_path = os.path.join(base_dir, "results")
os.makedirs(results_path, exist_ok=True)

# حفظ النتائج
edges_path = os.path.join(results_path, "edges.jpg")
thresh_path = os.path.join(results_path, "threshold.jpg")
cv2.imwrite(edges_path, edges)
cv2.imwrite(thresh_path, thresh)

print(f"✅ تم حفظ صورة الحواف في: {edges_path}")
print(f"✅ تم حفظ صورة Otsu's Thresholding في: {thresh_path}")

# جعل النوافذ قابلة للتحريك وإعادة التحجيم
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Edge Detection (Canny)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Otsu's Thresholding", cv2.WINDOW_NORMAL)

# عرض الصور
cv2.imshow("Original Image", image)
cv2.imshow("Edge Detection (Canny)", edges)
cv2.imshow("Otsu's Thresholding", thresh)

# انتظار أي مفتاح لإغلاق النوافذ
cv2.waitKey(0)
cv2.destroyAllWindows()
