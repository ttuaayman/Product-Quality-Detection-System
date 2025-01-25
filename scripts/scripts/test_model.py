import joblib
import pandas as pd

# تحميل النموذج المحسن
model_path = "models/fruit_classifier_improved.pkl"
model = joblib.load(model_path)

# بيانات اختبارية للتنبؤ (مع أسماء الميزات)
test_data = pd.DataFrame([[0.5, 0.3, 0.7, 0.8, 0.6, 0.2]], columns=[
    "WhitePixelRatio", "EdgePixelRatio", "Contrast", "Correlation", "Energy", "Homogeneity"
])

# استخدام النموذج للتنبؤ
prediction = model.predict(test_data)

# عرض النتيجة
print("🔍 Prediction:", prediction)
