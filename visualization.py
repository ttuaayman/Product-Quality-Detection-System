import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_train_val_accuracy(history_df, title="Training and Validation Accuracy", save_path=None):
    if not all(col in history_df.columns for col in ["epoch","accuracy","val_accuracy"]):
        print("تنبيه: الأعمدة (epoch, accuracy, val_accuracy) غير متوفرة في DataFrame.")
        return
    
    plt.figure(figsize=(8, 4))
    plt.plot(history_df['epoch'], history_df['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Val Accuracy', marker='s')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"تم حفظ منحنى الدقة في الملف: {save_path}")
    plt.show()

def plot_train_val_loss(history_df, title="Training and Validation Loss", save_path=None):
    if not all(col in history_df.columns for col in ["epoch","loss","val_loss"]):
        print("تنبيه: الأعمدة (epoch, loss, val_loss) غير متوفرة في DataFrame.")
        return
    
    plt.figure(figsize=(8, 4))
    plt.plot(history_df['epoch'], history_df['loss'], label='Train Loss', marker='o')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='s')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"تم حفظ منحنى الخسارة في الملف: {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    # تحقق من أن جميع التصنيفات موجودة في y_true
    unique_labels = np.unique(y_true)
    if not all(label in unique_labels for label in range(len(class_names))):
        print("تنبيه: بعض التصنيفات المحددة غير موجودة في y_true.")
        return
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"تم حفظ مصفوفة الارتباك في الملف: {save_path}")
    plt.show()

def show_classification_report(y_true, y_pred, class_names):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_sample_predictions(images, true_labels, pred_labels, class_names, samples=5, save_path=None):
    num_images = len(images)
    if num_images < samples:
        print(f"تحذير: عدد الصور ({num_images}) أقل من العينات المطلوبة ({samples}). سيتم عرض {num_images} فقط.")
        samples = num_images

    plt.figure(figsize=(samples * 3, 3))
    for i in range(samples):
        plt.subplot(1, samples, i+1)
        if isinstance(images[i], np.ndarray):
            plt.imshow(images[i].astype('uint8'))
        else:
            plt.imshow(images[i])
        
        if isinstance(true_labels[i], int):
            true_text = f"True: {class_names[true_labels[i]]}"
        else:
            true_text = f"True: {true_labels[i]}"

        if isinstance(pred_labels[i], int):
            pred_text = f"Pred: {class_names[pred_labels[i]]}"
        else:
            pred_text = f"Pred: {pred_labels[i]}"

        plt.title(f"{true_text}\n{pred_text}", fontsize=9)
        plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"تم حفظ عينات الصور في الملف: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("[INFO] بدء سكربت visualization.py ...")

    # تأكد من وجود مجلد النتائج
    os.makedirs("results", exist_ok=True)

    # مثال 1: قراءة سجل التدريب (CNN) من ملف csv
    cnn_history_file = "results/cnn_history.csv"
    if os.path.exists(cnn_history_file):
        print(f"[DEBUG] وجدنا ملف {cnn_history_file}, جارٍ القراءة...")
        cnn_history_df = pd.read_csv(cnn_history_file)
        plot_train_val_accuracy(cnn_history_df, title="CNN Training/Validation Accuracy", save_path="results/cnn_acc.png")
        plot_train_val_loss(cnn_history_df, title="CNN Training/Validation Loss", save_path="results/cnn_loss.png")
    else:
        print(f"[WARN] لم نجد الملف {cnn_history_file}، لن يتم رسم منحنيات الدقة والخسارة للـCNN")

    # مثال 2: قراءة نتائج التوقع (y_true, y_pred) لموديل CNN
    cnn_pred_file = "results/cnn_predictions.csv"
    if os.path.exists(cnn_pred_file):
        print(f"[DEBUG] وجدنا ملف {cnn_pred_file}, جارٍ القراءة...")
        pred_df = pd.read_csv(cnn_pred_file)
        if 'true_label' in pred_df.columns and 'pred_label' in pred_df.columns:
            y_true_cnn = pred_df['true_label'].values
            y_pred_cnn = pred_df['pred_label'].values
            class_names_cnn = ['apple_fresh','apple_rotten','banana_fresh','banana_rotten','no_fruit']
            plot_confusion_matrix(y_true_cnn, y_pred_cnn, class_names_cnn, title="CNN Confusion Matrix", save_path="results/cnn_cm.png")
            show_classification_report(y_true_cnn, y_pred_cnn, class_names_cnn)
        else:
            print(f"[WARN] ملف {cnn_pred_file} لا يحتوي الأعمدة 'true_label' و 'pred_label'")
    else:
        print(f"[WARN] لم نجد الملف {cnn_pred_file}، لن يتم رسم مصفوفة الارتباك للـCNN")

    # مثال 3: نفس المنطق لباقي النماذج (MobileNetV2)
    mobilenet_history_file = "results/mobilenetv2_history.csv"
    if os.path.exists(mobilenet_history_file):
        print(f"[DEBUG] وجدنا ملف {mobilenet_history_file}, جارٍ القراءة...")
        mobile_history_df = pd.read_csv(mobilenet_history_file)
        plot_train_val_accuracy(mobile_history_df, title="MobileNetV2 Training/Validation Accuracy", save_path="results/mobilenetv2_acc.png")
        plot_train_val_loss(mobile_history_df, title="MobileNetV2 Training/Validation Loss", save_path="results/mobilenetv2_loss.png")
    else:
        print(f"[WARN] لم نجد الملف {mobilenet_history_file}، لن يتم رسم منحنيات الدقة والخسارة لـMobileNetV2")

    mobilenet_pred_file = "results/mobilenetv2_predictions.csv"
    if os.path.exists(mobilenet_pred_file):
        print(f"[DEBUG] وجدنا ملف {mobilenet_pred_file}, جارٍ القراءة...")
        pred_df_mobilenet = pd.read_csv(mobilenet_pred_file)
        if 'true_label' in pred_df_mobilenet.columns and 'pred_label' in pred_df_mobilenet.columns:
            y_true_mobilenet = pred_df_mobilenet['true_label'].values
            y_pred_mobilenet = pred_df_mobilenet['pred_label'].values
            class_names_mobilenet = ['apple_fresh','apple_rotten','banana_fresh','banana_rotten','no_fruit']
            plot_confusion_matrix(y_true_mobilenet, y_pred_mobilenet, class_names_mobilenet, title="MobileNetV2 Confusion Matrix", save_path="results/mobilenetv2_cm.png")
            show_classification_report(y_true_mobilenet, y_pred_mobilenet, class_names_mobilenet)
        else:
            print(f"[WARN] ملف {mobilenet_pred_file} لا يحتوي الأعمدة 'true_label' و 'pred_label'")
    else:
        print(f"[WARN] لم نجد الملف {mobilenet_pred_file}، لن يتم رسم مصفوفة الارتباك لـMobileNetV2")

    # 4) أمثلة لعرض بعض الصور العشوائية:
    images_list = [np.random.rand(224, 224, 3) * 255 for _ in range(10)]
    true_list = np.random.randint(0, 5, size=10)
    pred_list = np.random.randint(0, 5, size=10)
    random_class_names = ['apple_fresh','apple_rotten','banana_fresh','banana_rotten','no_fruit']

    plot_sample_predictions(images_list, true_list, pred_list, random_class_names, samples=5, save_path="results/random_sample_preds.png")

    print("تم إكمال العرض المرئي للنتائج.")
