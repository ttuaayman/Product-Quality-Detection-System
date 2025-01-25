import pickle
import tensorflow as tf
from tensorflow import keras

# تحميل النموذج من ملف .pkl
with open("models/fruit_classifier_improved.pkl", "rb") as f:
    model_data = pickle.load(f)

# تأكد أن البيانات صالحة
if isinstance(model_data, keras.Model):
    model = model_data
    print("✅ Loaded Keras model successfully.")
else:
    print("❌ The loaded model is not a Keras model. Please check the pickle file.")
    exit()

# حفظ النموذج بصيغة h5
model.save("models/fruit_classifier.h5")
print("✅ Model saved as 'models/fruit_classifier.h5'")
