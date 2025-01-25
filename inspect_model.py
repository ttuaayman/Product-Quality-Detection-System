import pickle

# تحميل النموذج من ملف .pkl
with open("models/fruit_classifier_improved.pkl", "rb") as f:
    model = pickle.load(f)

# طباعة نوع البيانات داخل الملف
print(f"✅ Loaded object type: {type(model)}")

# إذا كان نموذج scikit-learn
if hasattr(model, "predict"):
    print("✅ This is a Scikit-learn model.")
else:
    print("❌ This is NOT a Scikit-learn model or Keras model.")
