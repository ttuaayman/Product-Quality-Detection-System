# AI-Based Fruit Classification System with Real-Time Quality Assessment

This project implements an AI-based Fruit Classification System that identifies fruit types and assesses their quality (e.g., fresh or rotten) through image analysis. It leverages deep learning—specifically Convolutional Neural Networks (CNNs) and Transfer Learning—to ensure accurate classification and quality detection. This solution is ideal for industries such as agriculture, retail, and e-commerce where product quality is critical.

---

## Features

- **Accurate Fruit Classification:** Identifies various fruit types and quality levels.
- **Real-Time Quality Assessment:** Detects whether a fruit is fresh or rotten.
- **Deep Learning & Transfer Learning:** Uses a standard CNN and MobileNetV2 for enhanced performance.
- **User Feedback Loop:** Captures user corrections to improve future model training.
- **Visualization Tools:** Generates confusion matrices, training curves, and classification reports.
<img width="1277" alt="Image" src="https://github.com/user-attachments/assets/abdc5a86-1dcd-4e84-a42e-5aa784a182d5" />
---

## New Additions and Modifications

1. **Multiple Training Scripts:**
   - **`train_cnn.py`:**  
     Trains a standard CNN on a 6-class dataset (e.g., `apple_fresh`, `apple_rotten`, `banana_fresh`, `banana_rotten`, `orange_fresh`, `orange_rotten`).
   - **`train_mobilenetv2.py`:**  
     Implements transfer learning using MobileNetV2 by freezing the initial layers and fine-tuning the final layers for the 6-class dataset.
   - **`train_model_final.py`:**  
     Combines CSV-based feature extraction with a small neural network and a RandomForest model for 6-class classification.
<img width="1270" alt="Image" src="https://github.com/user-attachments/assets/1c78287b-5bae-474f-ab98-6d1cc3ed0d22" />

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
<img width="1267" alt="Image" src="https://github.com/user-attachments/assets/2083f911-2ff4-435d-aa96-a48431eac012" />

---

## Directory Structure
```
Product-Quality-Detection/
│
├── app.py                # Main Flask application
├── train_cnn.py          # Script to train a custom CNN model
├── train_mobilenetv2.py  # Script to train a MobileNetV2 model (transfer learning)
├── train_model_final.py  # Another final training script (e.g., for RandomForest or improved CNN)
├── visualization.py      # Visualization script for plotting curves, confusion matrices, sample predictions
│
├── /static
│   ├── /css
│   │   └── style.css     # Styles for the web interface
│   ├── /images
│   │   ├── breda_robotics.png
│   │   └── utrecht_university.png
│   └── /js (if needed)
│
├── /templates
│   ├── index.html        # Homepage
│   ├── result.html       # Displays classification results
│   ├── dashboard.html    # Dashboard for real-time stats
│   ├── info.html         # Additional info pages
│   └── ...
│
├── /models
│   ├── fruit_classifier_cnn.h5
│   ├── fruit_classifier_mobilenetv2.h5
│   └── ...
│
├── /results
│   ├── cnn_history.csv           # Training logs (epochs, accuracy, loss, etc.) for CNN
│   ├── cnn_predictions.csv       # Predictions (true_label, pred_label) for CNN
│   ├── mobilenetv2_history.csv   # Training logs for MobileNetV2
│   ├── mobilenetv2_predictions.csv
│   ├── user_feedback.csv         # Stores user feedback from the web app
│   └── ...
│
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation (this file)
```

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
<img src="https://github.com/user-attachments/assets/770ac87e-76f2-41c5-a78d-1e5c44ba9def" alt="Image" width="250" />
<img src="https://github.com/user-attachments/assets/96d01af6-6360-45e8-9f12-03f2967c72a3" alt="Image" width="250" />
<img src="https://github.com/user-attachments/assets/2483af2f-a812-47b7-bed4-ac1f3e9bbcdb" alt="Image" width="150" />

<img src="https://github.com/user-attachments/assets/fd7c069c-491c-4e73-b03b-e20661fa2199" alt="Image" width="250" />
<img src="https://github.com/user-attachments/assets/7929ab31-31f4-4a0a-b71c-682537f87f4a" alt="Image" width="250" />
<img src="https://github.com/user-attachments/assets/52f63c81-3586-4f52-86dd-09b94396cdd1" alt="Image" width="150" />

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
