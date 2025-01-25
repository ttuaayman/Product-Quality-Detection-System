import pandas as pd
import matplotlib.pyplot as plt
import os

# 🟢 الحصول على المسار الأساسي للمشروع
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🟢 تحديد مسار ملف الميزات
features_csv = os.path.join(base_dir, "results", "features.csv")

# 🟢 التحقق مما إذا كان ملف الميزات موجودًا
if not os.path.exists(features_csv):
    print(f"❌ Features file not found: {features_csv}")
    exit()

# 🟢 تحميل البيانات من ملف CSV
df = pd.read_csv(features_csv)

# 🟢 عرض أول 5 صفوف من البيانات
print("\n🔍 **Preview of the first 5 rows of data:**")
print(df.head())

# 🟢 تحليل الإحصائيات الأساسية للميزات
print("\n📊 **Statistical Summary of Features:**")
print(df.describe())

# 🟢 رسم توزيع نسبة البكسلات البيضاء في Otsu's Thresholding
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df["WhitePixelRatio"], bins=20, color='blue', alpha=0.7)
plt.title("Distribution of White Pixel Ratio (Otsu's Thresholding)")
plt.xlabel("White Pixel Ratio")
plt.ylabel("Number of Images")

# 🟢 رسم توزيع نسبة الحواف المكتشفة في Canny Edge Detection
plt.subplot(1, 2, 2)
plt.hist(df["EdgePixelRatio"], bins=20, color='red', alpha=0.7)
plt.title("Distribution of Edge Pixel Ratio (Canny Edge Detection)")
plt.xlabel("Edge Pixel Ratio")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.show()
