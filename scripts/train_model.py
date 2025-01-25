import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🟢 الحصول على المسار الأساسي للمشروع
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🟢 تحديد مسار ملف الميزات
features_csv = os.path.join(base_dir, "results", "features.csv")

# 🟢 تحميل البيانات
df = pd.read_csv(features_csv)

# 🟢 تحويل التصنيفات (Fresh = 0, Rotten = 1)
df["Label"] = df["Label"].map({"fresh": 0, "rotten": 1})

# 🟢 اختيار الميزات والمتغير المستهدف
X = df[["WhitePixelRatio", "EdgePixelRatio"]]  # الميزات
y = df["Label"]  # الفئة المستهدفة

# 🟢 تقسيم البيانات إلى تدريب واختبار (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟢 إنشاء نموذج Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 🟢 تدريب النموذج
model.fit(X_train, y_train)

# 🟢 التنبؤ بالفئات على مجموعة الاختبار
y_pred = model.predict(X_test)

# 🟢 حساب دقة النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")

# 🟢 عرض تقرير التصنيف
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 🟢 التأكد من وجود مجلد "models/"
models_path = os.path.join(base_dir, "models")
os.makedirs(models_path, exist_ok=True)

# 🟢 حفظ النموذج
import joblib
model_path = os.path.join(models_path, "fruit_classifier.pkl")
joblib.dump(model, model_path)

print(f"\n✅ Model saved to: {model_path}")

