import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🟢 تحديد مسار البيانات المحسنة
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_csv = os.path.join(base_dir, "results", "features_improved.csv")

# 🟢 تحميل البيانات
df = pd.read_csv(features_csv)

# 🟢 تحويل التصنيفات إلى أرقام (Fresh = 0, Rotten = 1)
df["Label"] = df["Label"].map({"fresh": 0, "rotten": 1})

# 🟢 اختيار الميزات الجديدة والمتغير المستهدف
X = df[["WhitePixelRatio", "EdgePixelRatio", "Contrast", "Correlation", "Energy", "Homogeneity"]]
y = df["Label"]

# 🟢 تقسيم البيانات إلى تدريب (80%) واختبار (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟢 إنشاء نموذج Random Forest جديد
model = RandomForestClassifier(n_estimators=200, random_state=42)

# 🟢 تدريب النموذج
model.fit(X_train, y_train)

# 🟢 التنبؤ بالفئات على مجموعة الاختبار
y_pred = model.predict(X_test)

# 🟢 حساب دقة النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Improved Model Accuracy: {accuracy:.4f}")

# 🟢 عرض تقرير الأداء
print("\n📊 Improved Classification Report:")
print(classification_report(y_test, y_pred))

# 🟢 حفظ النموذج المحسن
models_path = os.path.join(base_dir, "models")
os.makedirs(models_path, exist_ok=True)
model_path = os.path.join(models_path, "fruit_classifier_improved.pkl")
joblib.dump(model, model_path)

print(f"\n✅ Improved Model saved to: {model_path}")
