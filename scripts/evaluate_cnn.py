import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Go one level up from "scripts" folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TEST_DIR = os.path.join(DATASET_DIR, "test_6class")  # or "test" if you renamed
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit_cnn.keras")

# Load the saved model
model = load_model(MODEL_PATH)

# Create a test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # keep order for confusion matrix
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions for confusion matrix
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
class_indices = test_generator.class_indices
sorted_class_indices = sorted(class_indices.items(), key=lambda x: x[1])
target_names = [k for k, v in sorted_class_indices]

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=target_names))
