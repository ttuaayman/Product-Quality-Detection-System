import cv2
import os

# تحديد نوع المنتج ونوع الجودة
product = "melon"  # اختر "apple" أو "banana" أو "melon"
quality = "fresh"  # اختر "fresh" أو "medium" أو "rotten"

# تحديد مسار حفظ الصور
save_path = f"dataset/{product}/{quality}/"
os.makedirs(save_path, exist_ok=True)  # إنشاء المجلد إذا لم يكن موجودًا

# فتح الكاميرا
cap = cv2.VideoCapture(0)  # استخدم 0 لكاميرا اللابتوب أو غيّره لمنفذ آخر

count = 0  # عداد الصور

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ لم يتم التقاط الصورة!")
        break

    # عرض الصورة الحية
    cv2.imshow("Camera View - اضغط 's' لحفظ الصورة", frame)

    # انتظار إدخال المستخدم
    key = cv2.waitKey(1)

    if key == ord('s'):  # عند الضغط على 's' يتم حفظ الصورة
        img_name = f"{save_path}/{product}_{quality}_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"✅ تم حفظ الصورة: {img_name}")
        count += 1

    elif key == ord('q'):  # عند الضغط على 'q' يتم إغلاق الكاميرا
        break

# إغلاق الكاميرا
cap.release()
cv2.destroyAllWindows()
