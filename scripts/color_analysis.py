import cv2
import numpy as np
import matplotlib.pyplot as plt

# تحميل الصورة
image_path = "dataset/sample.jpg"  # استخدم أي صورة من بياناتك
image = cv2.imread(image_path)

# التحقق من تحميل الصورة
if image is None:
    print("❌ لم يتم العثور على الصورة. تأكد من المسار الصحيح!")
    exit()

# تحويل الصورة من BGR إلى RGB (لأن OpenCV يستخدم BGR بشكل افتراضي)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# تقسيم الصورة إلى القنوات اللونية (الأحمر، الأخضر، الأزرق)
colors = ("red", "green", "blue")
channels = cv2.split(image_rgb)

# رسم الـ Histogram لكل قناة لونية
plt.figure(figsize=(10, 5))
plt.title("Color Histogram")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")

for channel, color in zip(channels, colors):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)

# عرض الـ Histogram
plt.show()
