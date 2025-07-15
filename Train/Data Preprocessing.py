import os
import cv2
import random
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Set Paths
BASE_PATH      = r"C:\Users\danie\PycharmProjects\FYP2025"
DATASET_PATH   = os.path.join(BASE_PATH, "Sipakmed_Dataset")
PROCESSED_PATH = os.path.join(BASE_PATH, "Sipakmed_Processed")
TRAIN_PATH     = os.path.join(BASE_PATH, "Sipakmed_Train")
VAL_PATH       = os.path.join(BASE_PATH, "Sipakmed_Val")
TEST_PATH      = os.path.join(BASE_PATH, "Sipakmed_Test")

# Create necessary directories
for path in [PROCESSED_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH]:
    os.makedirs(path, exist_ok=True)

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
random.seed(42)

# Preprocess and save (resize only)
def preprocess_and_save():
    for class_folder in os.listdir(DATASET_PATH):
        class_input = os.path.join(DATASET_PATH, class_folder)
        if not os.path.isdir(class_input):
            continue
        class_out = os.path.join(PROCESSED_PATH, class_folder)
        os.makedirs(class_out, exist_ok=True)
        for fname in tqdm(os.listdir(class_input), desc=f"Processing {class_folder}"):
            if not any(fname.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                continue
            img_path = os.path.join(class_input, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # resize to 224x224
            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            # save
            out_path = os.path.join(class_out, fname)
            cv2.imwrite(out_path, resized)

# Split processed data into train/val/test
def split_dataset():
    for class_folder in os.listdir(PROCESSED_PATH):
        class_input = os.path.join(PROCESSED_PATH, class_folder)
        images = [os.path.join(class_input, f) for f in os.listdir(class_input)]
        random.shuffle(images)
        train_val, test = train_test_split(images, test_size=0.3, random_state=42)
        train, val      = train_test_split(train_val, test_size=0.2, random_state=42)
        for subset, target_root in [(train, TRAIN_PATH), (val, VAL_PATH), (test, TEST_PATH)]:
            out_dir = os.path.join(target_root, class_folder)
            os.makedirs(out_dir, exist_ok=True)
            for src in subset:
                shutil.copy(src, out_dir)

# Run Full Pipeline
def prepare_sipakmed_pipeline():
    print("🔧 Resizing images to 224×224...")
    preprocess_and_save()
    print("📦 Splitting into Train/Val/Test...")
    split_dataset()
    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    prepare_sipakmed_pipeline()

