import numpy as np
import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ تحميل البيانات من ملف `pkl`
with open("models/fruit_classifier_improved.pkl", "rb") as f:
    data = pickle.load(f)

# 2️⃣ التحقق من نوع البيانات
if isinstance(data, np.ndarray):
    print(f"✅ Loaded data shape: {data.shape}")

    # ✅ تحويل البيانات إلى `DataFrame` لفحصها
    df = pd.DataFrame(data)

    # 3️⃣ عرض أول 5 صفوف للتحقق من القيم
    print("\n🔍 Sample Data (Before Cleaning):")
    print(df.head())

    # 4️⃣ إذا كانت البيانات تحتوي على أسماء ميزات، نقوم بحذفها
    if isinstance(df.iloc[0, 0], str):  # إذا كان الصف الأول يحتوي على نصوص، فهذه أسماء ميزات
        print("❌ Data contains feature names instead of values. Removing the first row...")
        df = df.iloc[1:].reset_index(drop=True)  # حذف الصف الأول وإعادة فهرسة البيانات

    # 5️⃣ التأكد من تحويل جميع القيم إلى أرقام
    df = df.apply(pd.to_numeric, errors="coerce")  # تحويل النصوص إلى أرقام مع استبدال القيم غير القابلة للتحويل بـ NaN
    df = df.dropna()  # إزالة أي صفوف تحتوي على `NaN`
    
    # ✅ التحقق من عدد الصفوف بعد التنظيف
    print(f"\n📊 Data shape after cleaning: {df.shape}")

    # 🔥 **إذا لم يتبق أي بيانات بعد التنظيف، إيقاف البرنامج**
    if df.shape[0] == 0:
        print("❌ No valid data available after cleaning! Check the original dataset.")
        exit()

    # 6️⃣ استخراج البيانات بعد التصحيح
    X = df.values  # تحويل `DataFrame` إلى `numpy`
    y = np.random.randint(0, 2, size=X.shape[0])  # **يجب استبدال هذه القيم بتسميات صحيحة**

    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7️⃣ إنشاء نموذج `SVM` وتدريبه
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    # 8️⃣ تقييم النموذج
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Retrained Model Accuracy: {accuracy:.4f}")

    # طباعة تقرير التصنيف
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 9️⃣ حفظ النموذج الجديد بصيغة `pkl`
    with open("models/fruit_classifier_sklearn.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n✅ Model successfully retrained and saved as 'models/fruit_classifier_sklearn.pkl'.")

else:
    print("❌ The loaded data is not valid for training.")
