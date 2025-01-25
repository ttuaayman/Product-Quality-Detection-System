import pickle

# مسار النموذج المخزن
PKL_MODEL_PATH = "models/fruit_classifier.pkl"

# تحميل البيانات من الملف
with open(PKL_MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# طباعة نوع البيانات المحملة
print(f"✅ Loaded object type: {type(model)}")

# إذا كان النموذج `Keras`
if hasattr(model, "save"):
    print("✅ This is a Keras model.")
elif hasattr(model, "predict"):
    print("✅ This is likely a Scikit-learn model.")
else:
    print("❌ The file does not contain a recognizable model type.")
