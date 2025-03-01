import os
import shutil

# Arabic explanation is outside, English #comments are inside.

# Define the original base directory
BASE_DIR = "dataset"

# We assume you have 'train' and 'test' subfolders with apple/banana/orange => fresh/rotten
SPLITS = ["train", "test"]
FRUITS = ["apple", "banana", "orange"]
QUALITIES = ["fresh", "rotten"]

# Temporary folders to move the new structure
TRAIN_6CLASS = os.path.join(BASE_DIR, "train_6class")
TEST_6CLASS = os.path.join(BASE_DIR, "test_6class")

os.makedirs(TRAIN_6CLASS, exist_ok=True)
os.makedirs(TEST_6CLASS, exist_ok=True)

def move_to_6class(old_split_dir, new_split_dir):
    """
    old_split_dir might be 'dataset/train' or 'dataset/test'
    new_split_dir might be 'dataset/train_6class' or 'dataset/test_6class'
    """
    for fruit in FRUITS:
        for quality in QUALITIES:
            # Old path: e.g. dataset/train/apple/fresh
            old_path = os.path.join(old_split_dir, fruit, quality)
            
            # New folder name: e.g. apple_fresh
            new_folder_name = f"{fruit}_{quality}"
            new_path = os.path.join(new_split_dir, new_folder_name)
            
            os.makedirs(new_path, exist_ok=True)
            
            # Move all files from old_path to new_path
            if os.path.exists(old_path):
                for filename in os.listdir(old_path):
                    old_file = os.path.join(old_path, filename)
                    if os.path.isfile(old_file):
                        shutil.move(old_file, os.path.join(new_path, filename))
            else:
                print(f"Warning: {old_path} does not exist. Skipping...")

# Move train data to train_6class
move_to_6class(os.path.join(BASE_DIR, "train"), TRAIN_6CLASS)

# Move test data to test_6class
move_to_6class(os.path.join(BASE_DIR, "test"), TEST_6CLASS)

# Optional: remove old 'train' and 'test' folders if you want
old_train_dir = os.path.join(BASE_DIR, "train")
old_test_dir = os.path.join(BASE_DIR, "test")
if os.path.exists(old_train_dir):
    shutil.rmtree(old_train_dir)
if os.path.exists(old_test_dir):
    shutil.rmtree(old_test_dir)

# Rename train_6class -> train
os.rename(TRAIN_6CLASS, old_train_dir)
# Rename test_6class -> test
os.rename(TEST_6CLASS, old_test_dir)

print("✅ Dataset reorganized successfully!")
print("Now you should have 6 folders in 'train' and 6 folders in 'test'.")
