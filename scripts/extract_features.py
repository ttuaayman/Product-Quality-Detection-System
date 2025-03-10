import cv2
import numpy as np
import os
from skimage.feature import greycomatrix, greycoprops  # أو graycomatrix, graycoprops
from skimage import img_as_ubyte

# المسار الأساسي للمشروع
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# تحديد المسار الصحيح للصورة
image_path = os.path.join(base_dir, "dataset", "sample.jpg")

# التحقق من وجود الصورة
if not os.path.exists(image_path):
    print(f"❌ لم يتم العثور على الصورة في المسار: {image_path}")
    print("🔹 تأكد من أن الصورة موجودة داخل مجلد 'dataset/' واسمها 'sample.jpg'.")
    exit()

# تحميل الصورة
image = cv2.imread(image_path)

# تحويل الصورة إلى تدرج الرمادي
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# تطبيق Otsu's Thresholding
_, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# حساب عدد البكسلات البيضاء
white_pixels = np.sum(thresh == 255)
total_pixels = thresh.shape[0] * thresh.shape[1]
white_pixel_ratio = white_pixels / total_pixels

# تطبيق Canny Edge Detection
edges = cv2.Canny(gray_image, 80, 200)

# حساب عدد البكسلات البيضاء في صورة Canny
edge_pixels = np.sum(edges == 255)
edge_pixel_ratio = edge_pixels / total_pixels

# استخراج GLCM (Gray Level Co-occurrence Matrix)
glcm = greycomatrix(img_as_ubyte(gray_image), [1], [0], symmetric=True, normed=True)

# حساب التباين، الارتباط، والطاقة
contrast = greycoprops(glcm, 'contrast')[0, 0]
correlation = greycoprops(glcm, 'correlation')[0, 0]
energy = greycoprops(glcm, 'energy')[0, 0]

# حساب متوسط شدة اللون
mean_intensity = np.mean(gray_image) / 255

# حساب التباين في الصورة
variance = np.var(gray_image) / 255

# طباعة الميزات المستخرجة
print("\n🔍 **استخراج الميزات من الصورة:**")
print(f"✅ نسبة البكسلات البيضاء في Otsu's Thresholding: {white_pixel_ratio:.4f}")
print(f"✅ نسبة الحواف المكتشفة في Canny Edge Detection: {edge_pixel_ratio:.4f}")
print(f"✅ التباين (Contrast): {contrast:.4f}")
print(f"✅ الارتباط (Correlation): {correlation:.4f}")
print(f"✅ الطاقة (Energy): {energy:.4f}")
print(f"✅ متوسط شدة اللون (Mean Intensity): {mean_intensity:.4f}")
print(f"✅ التباين في الصورة (Variance): {variance:.4f}")

# التأكد من وجود مجلد النتائج
results_path = os.path.join(base_dir, "results")
os.makedirs(results_path, exist_ok=True)

# حفظ النتائج
cv2.imwrite(os.path.join(results_path, "threshold.jpg"), thresh)
cv2.imwrite(os.path.join(results_path, "edges.jpg"), edges)

print(f"✅ تم حفظ النتائج في مجلد 'results/'.")

