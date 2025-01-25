import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore

# 🟢 تحديد مسار البيانات المحسنة
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_csv = os.path.join(base_dir, "results", "features_5improved.csv")

# 🟢 تحميل البيانات
df = pd.read_csv(features_csv)

# 🟢 تحويل التصنيفات إلى أرقام (Fresh = 0, Rotten = 1)
df["Label"] = df["Label"].map({"fresh": 0, "rotten": 1})

# 🟢 اختيار 6 ميزات والمتغير المستهدف
X = df[["WhitePixelRatio", "EdgePixelRatio", "Contrast", "Correlation", "Energy", "MeanIntensity"]]  # 6 ميزات
y = df["Label"]

# 🟢 تقسيم البيانات إلى تدريب (80%) واختبار (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟢 إنشاء نموذج Sequential جديد
model = Sequential()
model.add(Dense(64, input_dim=6, activation='relu'))  # Update input_dim to 6
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🟢 تدريب النموذج
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

# 🟢 حفظ النموذج
models_path = os.path.join(base_dir, "models")
os.makedirs(models_path, exist_ok=True)
model_path = os.path.join(models_path, "model_5improved.h5")
model.save(model_path)

# 🟢 إنشاء نموذج Random Forest جديد
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

# 🟢 تدريب النموذج
rf_model.fit(X_train, y_train)

# 🟢 التنبؤ بالفئات على مجموعة الاختبار
y_pred = rf_model.predict(X_test)

# 🟢 حساب دقة النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Improved Model Accuracy with 6 features: {accuracy:.4f}")

# 🟢 عرض تقرير الأداء
print("\n📊 Improved Classification Report:")
print(classification_report(y_test, y_pred))

# 🟢 حفظ النموذج المحسن
models_path = os.path.join(base_dir, "models")
os.makedirs(models_path, exist_ok=True)
model_path = os.path.join(models_path, "fruit_classifier_improved_5features.pkl")
joblib.dump(rf_model, model_path)

print(f"\n✅ Improved Model saved to: {model_path}")
