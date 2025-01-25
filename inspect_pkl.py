import pickle
import numpy as np

# مسار ملف النموذج
PKL_MODEL_PATH = "models/fruit_classifier.pkl"

# تحميل البيانات من الملف
with open(PKL_MODEL_PATH, "rb") as file:
    data = pickle.load(file)

# طباعة نوع البيانات
print(f"✅ Loaded object type: {type(data)}")

# إذا كان مصفوفة NumPy
if isinstance(data, np.ndarray):
    print(f"✅ NumPy array shape: {data.shape}")
    print(f"✅ NumPy array preview: {data[:5]}")  # طباعة أول 5 قيم لفهم المحتوى
else:
    print("❌ Unexpected data type in the pickle file.")
