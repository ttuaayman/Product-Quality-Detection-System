import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# تحميل النموذج من ملف .pkl
pkl_path = "models/fruit_classifier_improved.pkl"

with open(pkl_path, "rb") as f:
    sk_model = pickle.load(f)

# التأكد أن الملف يحتوي على نموذج `Scikit-learn`
if hasattr(sk_model, "predict"):
    print("✅ This is a Scikit-learn model. Converting to Keras...")

    # التحقق مما إذا كان النموذج يستخدم SVM
    if hasattr(sk_model, "support_vectors_") and hasattr(sk_model, "dual_coef_"):
        X_train = sk_model.support_vectors_
        y_train = sk_model.dual_coef_.ravel()

        # إنشاء نموذج `Keras` مشابه
        model = Sequential([
            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        # تجميع النموذج
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # إعادة تدريب النموذج في `Keras`
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # حفظ النموذج بصيغة `.h5`
        keras_model_path = "models/fruit_classifier.h5"
        model.save(keras_model_path)
        print(f"✅ Model successfully converted to Keras and saved as '{keras_model_path}'.")

    else:
        print("❌ The Scikit-learn model is not an SVM or does not contain the expected attributes.")
else:
    print("❌ The loaded model is not a Scikit-learn model.")
