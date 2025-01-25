import pickle
import numpy as np

# تحميل البيانات
with open("models/fruit_classifier_improved.pkl", "rb") as f:
    data = pickle.load(f)

print(f"✅ Loaded object type: {type(data)}")

# إذا كان `numpy.ndarray`
if isinstance(data, np.ndarray):
    print(f"✅ NumPy array shape: {data.shape}")

    # 🔍 **عرض البيانات الأولية**
    print(f"🔍 First 5 rows:\n{data[:5]}")

    # التحقق مما إذا كانت البيانات تحتوي على أرقام فقط
    if np.issubdtype(data.dtype, np.number):
        print("✅ The dataset contains numerical values.")
    else:
        print("❌ The dataset contains non-numerical values!")
        print("⚠ Removing non-numerical rows...")

        # **إزالة الصف الأول لأنه يحتوي على أسماء الميزات**
        try:
            data = np.array(data[1:], dtype=np.float32)
            print("✅ Successfully removed non-numeric values.")
            print(f"🔍 Cleaned Data Sample:\n{data[:5]}")
        except ValueError as e:
            print(f"❌ Failed to convert data: {e}")
            exit()

    # حفظ البيانات المنظفة
    with open("models/cleaned_fruit_data.pkl", "wb") as f:
        pickle.dump(data, f)

    print("✅ Cleaned data saved as 'models/cleaned_fruit_data.pkl'.")

else:
    print("❌ The loaded data is not in the expected format.")
