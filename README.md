# AI-Based Fruit Classification System with Real-Time Quality Assessment

This project implements an AI-based Fruit Classification System that identifies fruit types and assesses their quality (e.g., fresh or rotten) through image analysis. It leverages deep learning—specifically Convolutional Neural Networks (CNNs) and Transfer Learning—to ensure accurate classification and quality detection. This solution is ideal for industries such as agriculture, retail, and e-commerce where product quality is critical.

---

## Features

- **Accurate Fruit Classification:** Identifies various fruit types and quality levels.
- **Real-Time Quality Assessment:** Detects whether a fruit is fresh or rotten.
- **Deep Learning & Transfer Learning:** Uses a standard CNN and MobileNetV2 for enhanced performance.
- **User Feedback Loop:** Captures user corrections to improve future model training.
- **Visualization Tools:** Generates confusion matrices, training curves, and classification reports.

---

## New Additions and Modifications

1. **Multiple Training Scripts:**
   - **`train_cnn.py`:**  
     Trains a standard CNN on a 6-class dataset (e.g., `apple_fresh`, `apple_rotten`, `banana_fresh`, `banana_rotten`, `orange_fresh`, `orange_rotten`).
   - **`train_mobilenetv2.py`:**  
     Implements transfer learning using MobileNetV2 by freezing the initial layers and fine-tuning the final layers for the 6-class dataset.
   - **`train_model_final.py`:**  
     Combines CSV-based feature extraction with a small neural network and a RandomForest model for 6-class classification.

2. **Enhanced Visualization and Analysis:**
   - **Confusion Matrix:**  
     Visual representations (for both CNN and MobileNetV2) show correct vs. incorrect predictions.
   - **Classification Report:**  
     Provides precision, recall, F1-score, and support for each class.
   - **Loss and Accuracy Curves:**  
     Charts tracking training and validation accuracy and loss over epochs.
   - **Sample Predictions:**  
     A new image (`random_sample_preds.png`) displays model performance on random test images.

3. **Thresholding Logic for “No Fruit”:**
   - In `app.py`, a confidence threshold (e.g., `THRESHOLD = 0.7`) is used to return "No Fruit Detected" if the highest probability is below the set threshold.

4. **Dashboard and Feedback Mechanism:**
   - **User Feedback:**  
     Allows users to provide corrections (stored in `user_feedback.csv`), enabling future model improvements.
   - **Dashboard Updates:**  
     The `/dashboard` route now displays additional categories, including a “No Fruit” percentage.

---

## Directory Structure

Product-Quality-Detection/ │ ├── app.py # Main Flask application ├── train_cnn.py # Script to train a standard CNN ├── train_mobilenetv2.py # Script to train MobileNetV2 with transfer learning ├── train_model_final.py # Combined approach using CSV-based features and ML ├── results/ # Contains logs, confusion matrices, training curves, classification reports, and sample predictions │ ├── cnn_acc.png │ ├── cnn_loss.png │ ├── cnn_cm.png │ ├── mobilenetv2_acc.png │ ├── mobilenetv2_loss.png │ ├── mobilenetv2_cm.png │ ├── random_sample_preds.png │ └── user_feedback.csv ├── models/ # Saved models (e.g., fruit_cnn.keras, fruit_mobilenetv2.keras, etc.) ├── dataset/ # Datasets for training and testing (train_6class, test_6class) ├── static/ # Static files (CSS, images) ├── templates/ # HTML templates (index.html, result.html, dashboard.html, etc.) ├── requirements.txt # Python dependencies └── README.md # This file

yaml
Copy

---

## Installation and Usage

### 1. Install Dependencies
```
pip install -r requirements.txt
```
2. Train the Models
Train CNN Model:
```python train_cnn.py```
Trains a standard CNN using images from dataset/train_6class.
Saves the model as models/fruit_cnn.keras.
Train MobileNetV2 Model:
```python train_mobilenetv2.py```
Uses MobileNetV2 (with include_top=False), freezes initial layers, fine-tunes final layers.
Saves the model as models/fruit_mobilenetv2.keras.
3. Deploy the Web Application
Start the Flask server:
```python app.py```
Open your browser and navigate to http://127.0.0.1:5000.
You can upload images or capture them via the camera.
The system will classify the fruit and its quality, or display “No Fruit Detected” if the confidence is below the threshold.
4. Analyze the Results
- Confusion Matrices & Training Curves:
Located in the results/ folder.
- Classification Reports:
Printed in the terminal and saved as log files.
- Sample Predictions:
Check the random_sample_preds.png image for an overview of test predictions.
## Explanation of Figures and Comparisons
CNN Model:
Accuracy & Loss Curves:
cnn_acc.png and cnn_loss.png show the evolution of training and validation metrics over epochs. The training accuracy typically increases more rapidly, indicating model fitting.
MobileNetV2 Model:
Accuracy & Loss Curves:
mobilenetv2_acc.png and mobilenetv2_loss.png reflect faster convergence and improved validation accuracy due to transfer learning.
Confusion Matrix:
Visualizes the number of correct and misclassified predictions across all classes.
Sample Predictions:
The random_sample_preds.png image displays random test images along with their true and predicted labels. If the confidence threshold is not met, “No Fruit” is indicated.
## Future Improvements
Expand Classes:
Add more fruit types (e.g., mango, grape) or additional quality levels (e.g., medium, overripe).
Data Augmentation:
Incorporate more image transformations to handle background variations and lighting conditions.
Enhanced Feedback Loop:
Utilize data from user_feedback.csv for continuous model re-training or fine-tuning.
Edge Deployment:
Optimize the MobileNetV2 model for deployment on mobile or embedded devices (e.g., Raspberry Pi).
