import os

# المسار الأساسي للبيانات
dataset_path = "dataset/train/"

# المنتجات الموجودة في مجموعة البيانات
categories = ["apple", "banana", "orange"]  # استبدلنا "melon" بـ "orange"
labels = ["fresh", "rotten"]  # حذفنا "medium"

print("\n🔍 **تحليل عدد الصور في كل فئة:**")
# حساب عدد الصور في كل فئة
for category in categories:
    print(f"\n📂 {category.upper()} DATA:")
    for label in labels:
        path = os.path.join(dataset_path, category, label)
        if os.path.exists(path):
            num_images = len(os.listdir(path))
            print(f"  - {label}: {num_images} صور")
        else:
            print(f"  - {label}: ❌ لا يوجد مجلد")
