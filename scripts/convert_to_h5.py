import joblib
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# تحميل النموذج
model_path = "models/fruit_classifier_improved.pkl"
sk_model = joblib.load(model_path)

# التحقق من أن النموذج من نوع Scikit-learn
if hasattr(sk_model, "predict"):
    print("✅ Scikit-learn model loaded. Converting to Keras...")

    # إنشاء نموذج Keras جديد
    keras_model = Sequential([
        Dense(64, activation="relu", input_dim=6),  # استبدل `6` بعدد الميزات لديك
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    # تجميع النموذج
    keras_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # بيانات تدريب تجريبية (استبدلها ببياناتك الفعلية)
    X_train = [[0.5, 0.3, 0.7, 0.8, 0.6, 0.2]]  # تأكد من أن بياناتك هنا صحيحة
    y_train = [1]

    # تحويل البيانات إلى مصفوفات NumPy
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("⚠ Retraining the Keras model...")
    keras_model.fit(X_train, y_train, epochs=10, batch_size=1)

    # حفظ النموذج بصيغة H5
    keras_model.save("models/fruit_classifier.h5")
    print("✅ Model successfully converted and saved to 'models/fruit_classifier.h5'.")

else:
    print("❌ The loaded model is not a Scikit-learn model.")
