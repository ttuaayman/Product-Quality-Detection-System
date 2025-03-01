import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 🔹 تحديد المسارات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(BASE_DIR, "results", "features_5improved.csv")  # تأكد من أن اسم الملف صحيح
MODELS_PATH = os.path.join(BASE_DIR, "models")
MODEL_H5_PATH = os.path.join(MODELS_PATH, "model_6class.h5")
MODEL_RF_PATH = os.path.join(MODELS_PATH, "fruit_classifier_6classes.pkl")

# 🔹 التأكد من وجود المسارات المطلوبة
os.makedirs(MODELS_PATH, exist_ok=True)

# 🔹 تحميل بيانات التدريب من ملف CSV
df = pd.read_csv(FEATURES_CSV)

# التحقق من وجود العمود 'FruitType'
if 'FruitType' not in df.columns:
    raise KeyError("Column 'FruitType' not found in the dataset")

# ✅ **تحويل جودة الفاكهة إلى أرقام**
df["Label"] = df["Label"].map({
    "fresh": 0,
    "rotten": 1
})

# ✅ **دمج الفاكهة والجودة في تصنيف واحد**
df["Class"] = df["FruitType"] * 2 + df["Label"]  # (0=Apple_Fresh, 1=Apple_Rotten, 2=Banana_Fresh, ...)

# ✅ **اختيار الميزات والمتغير المستهدف**
X = df[["WhitePixelRatio", "EdgePixelRatio", "Contrast", "Correlation", "Energy", "MeanIntensity"]]
y = df["Class"]

# ✅ **تحويل y إلى صيغة One-Hot Encoding**
y = to_categorical(y, num_classes=6)

# ✅ **تقسيم البيانات إلى تدريب (80%) واختبار (20%)**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================== #
# ✅ **إضافة `ImageDataGenerator` لتحميل بيانات التدريب والتحقق** #
# ============================================================== #

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 🔹 تحميل بيانات التدريب
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
train_generator = datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# 🔹 تحميل بيانات التحقق
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
validation_generator = datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("\n✅ Data Generators Loaded Successfully.")

# ============================================================== #
# ✅ **بناء نموذج `Sequential` (شبكة عصبية) لتصنيف 6 فئات**      #
# ============================================================== #

model = Sequential([
    Dense(128, input_dim=6, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # 6 فئات بدلاً من 2
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ **تدريب النموذج**
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# ✅ **حفظ النموذج**
model.save(MODEL_H5_PATH)
print(f"\n✅ Neural Network Model saved to: {MODEL_H5_PATH}")

# ============================================================== #
# ✅ **بناء نموذج `RandomForestClassifier` وتحسينه لـ 6 فئات**    #
# ============================================================== #

rf_model = RandomForestClassifier(n_estimators=500, random_state=42)

# ✅ **تدريب النموذج**
rf_model.fit(X_train, np.argmax(y_train, axis=1))  # تحويل One-Hot إلى تصنيفات

# ✅ **التنبؤ بالفئات على مجموعة الاختبار**
y_pred = rf_model.predict(X_test)

# ✅ **حساب دقة النموذج**
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
print(f"\n✅ Improved Model Accuracy with 6 classes: {accuracy:.4f}")

# ✅ **عرض تقرير الأداء**
print("\n📊 Improved Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred))

# ✅ **حفظ النموذج المحسن**
joblib.dump(rf_model, MODEL_RF_PATH)
print(f"\n✅ Improved Model saved to: {MODEL_RF_PATH}")
