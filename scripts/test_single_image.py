import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 1) تحديد المسار للنموذج
model_path = "c:/Product-Quality-Detection/models/fruit_mobilenetv2.keras"
model = load_model(model_path)

# 2) تحديد أسماء الفئات بالترتيب نفسه الذي يستخدمه ImageDataGenerator
# إذا كان ترتيبك الأبجدي كالتالي:
# apple_fresh (0), apple_rotten (1), banana_fresh (2), banana_rotten (3), orange_fresh (4), orange_rotten (5)
class_names = [
    "apple_fresh",
    "apple_rotten",
    "banana_fresh",
    "banana_rotten",
    "orange_fresh",
    "orange_rotten"
]

# 3) تحديد مسار الصورة التي تريد اختبارها
test_image_path = "path/to/any_image.jpg"

# 4) تحميل الصورة بالحجم المطلوب (224x224 إذا كنت تستخدم MobileNetV2)
img = image.load_img(test_image_path, target_size=(224, 224))

# 5) تحويل الصورة إلى مصفوفة NumPy
img_array = image.img_to_array(img)

# 6) توسيع الأبعاد لتصبح [1, 224, 224, 3]
img_array = np.expand_dims(img_array, axis=0)

# 7) إعادة قياس (Normalize) القيم إلى [0..1] إذا كان النموذج تدرب على 1./255
img_array = img_array / 255.0

# 8) استدعاء model.predict
predictions = model.predict(img_array)

# 9) استخراج أعلى احتمال
predicted_class_idx = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_idx]

predicted_class_name = class_names[predicted_class_idx]

print(f"Predicted class: {predicted_class_name} (Confidence: {confidence:.3f})")
