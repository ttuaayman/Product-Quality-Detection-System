import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Arabic explanation is outside, English #comments are inside this code.

# 1) Define base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# If your folders are named "train_6class" and "test_6class", define them here:
TRAIN_DIR = os.path.join(DATASET_DIR, "train_6class")
TEST_DIR = os.path.join(DATASET_DIR, "test_6class")

# Create a directory to save the model
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 2) Create an ImageDataGenerator for training/validation with 20% validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training generator uses subset='training'
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation generator uses subset='validation'
validation_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 3) Build a simple CNN model
model = Sequential()

# First Convolution + Pooling
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution + Pooling
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten + Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # dropout to reduce overfitting
model.add(Dense(6, activation='softmax'))  # 6 classes (apple_fresh, apple_rotten, banana_fresh, banana_rotten, orange_fresh, orange_rotten)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4) Train the model
EPOCHS = 20
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 5) Save the model
model_path = os.path.join(MODELS_DIR, "fruit_cnn.keras")
model.save(model_path)
print(f"Model saved to: {model_path}")

# 6) Evaluate on the test set (test_6class)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
