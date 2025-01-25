import numpy as np
import os
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# 🟢 تحديد مسار البيانات المحسنة
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_csv = os.path.join(base_dir, "results", "features_improved.csv")

# 🟢 تحميل البيانات
import pandas as pd
df = pd.read_csv(features_csv)

# 🟢 تحويل التصنيفات إلى أرقام (Fresh = 0, Rotten = 1)
df["Label"] = df["Label"].map({"fresh": 0, "rotten": 1})

# 🟢 استخراج الميزات
def extract_features(image_path):
    """ استخراج الميزات من الصورة """
    img = Image.open(image_path)
    img = img.convert("L")  # تحويل الصورة إلى تدرجات الرمادي
    img = np.array(img)

    # استخدم طريقة استخراج الخصائص مثل النسبة البيكسل البيضاء أو الخصائص الأخرى
    white_pixel_ratio = np.sum(img > 200) / img.size  # نسبة البكسلات البيضاء
    edge_pixel_ratio = np.sum(np.gradient(img)) / img.size  # مثال على استخراج حافة الصورة
    contrast = greycoprops(greycomatrix(img, [1], [0]), 'contrast')[0, 0]  # استخدام GLCM لاستخراج التباين
    correlation = greycoprops(greycomatrix(img, [1], [0]), 'correlation')[0, 0]  # استخراج الارتباط
    energy = greycoprops(greycomatrix(img, [1], [0]), 'energy')[0, 0]  # استخراج الطاقة

    # العودة بـ 5 ميزات فقط (بدون التماثل)
    return np.array([white_pixel_ratio, edge_pixel_ratio, contrast, correlation, energy])

# 🟢 استخراج الميزات الجديدة والمتغير المستهدف
X = df[["WhitePixelRatio", "EdgePixelRatio", "Contrast", "Correlation", "Energy"]]
y = df["Label"]

# 🟢 تقسيم البيانات إلى تدريب (80%) واختبار (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟢 إنشاء نموذج Keras مع 5 ميزات فقط
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),  # هنا نحدد المدخلات بـ 5 بدلاً من 6
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # لأننا نتعامل مع 3 تصنيفات (فواكه)
])

# 🟢 تجميع وتدريب النموذج
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 🟢 حفظ النموذج المحسن
models_path = os.path.join(base_dir, "models")
os.makedirs(models_path, exist_ok=True)
model_path = os.path.join(models_path, "fruit_classifier_improved_model.h5")
model.save(model_path)

print(f"✅ Improved Model saved to: {model_path}")
