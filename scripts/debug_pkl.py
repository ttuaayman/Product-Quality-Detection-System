import pickle

# تحميل البيانات من الملف
with open("models/fruit_classifier_improved.pkl", "rb") as f:
    data = pickle.load(f)

print(f"✅ Loaded object type: {type(data)}")

# إذا كان `numpy.ndarray`
if isinstance(data, (list, tuple)):
    print(f"✅ Data length: {len(data)}")
    print(f"🔍 First 5 entries: {data[:5]}")

elif isinstance(data, dict):
    print("✅ Data is a dictionary. Keys:")
    for key in data.keys():
        print(f"  - {key}")

elif isinstance(data, str):
    print("❌ Data is stored as a string!")

else:
    print("⚠ Unknown data format.")

