import os
import shutil
import cv2
import random

# Paths
raw_data_dir = "./raw_data"
output_dirs = {
    "train": {"images": "./data_train/images", "masks": "./data_train/masks"},
    "valid": {"images": "./data_valid/images", "masks": "./data_valid/masks"},
    "test":  {"images": "./data_test/images",  "masks": "./data_test/masks"},
}

target_size = (300, 300)  # resize images and masks
split_ratios = {"train": 0.7, "valid": 0.15, "test": 0.15}

# Make directories
for split in output_dirs:
    os.makedirs(output_dirs[split]["images"], exist_ok=True)
    os.makedirs(output_dirs[split]["masks"], exist_ok=True)

# Collect all image filenames
all_images = []
all_masks = []

for root, dirs, files in os.walk(raw_data_dir):
    for file in files:
        if not file.lower().endswith(".png"):
            continue
        file_path = os.path.join(root, file)
        if "mask" in root.lower():
            all_masks.append(file_path)
        else:
            all_images.append(file_path)

# Ensure images and masks match
all_images.sort()
all_masks.sort()
assert len(all_images) == len(all_masks), "Number of images and masks must match"

# Shuffle dataset
combined = list(zip(all_images, all_masks))
random.shuffle(combined)
all_images, all_masks = zip(*combined)

# Compute split indices
n = len(all_images)
train_idx = int(split_ratios["train"] * n)
valid_idx = int((split_ratios["train"] + split_ratios["valid"]) * n)

splits = {
    "train": (0, train_idx),
    "valid": (train_idx, valid_idx),
    "test":  (valid_idx, n)
}

# Process and save
for split, (start, end) in splits.items():
    for img_path, mask_path in zip(all_images[start:end], all_masks[start:end]):
        # Process image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv2.resize(img, target_size)

        # Process mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Save files
        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)

        cv2.imwrite(os.path.join(output_dirs[split]["images"], img_name), img)
        cv2.imwrite(os.path.join(output_dirs[split]["masks"], mask_name), mask)

print("Preprocessing and split complete!")

"""import os
import shutil
import cv2

raw_data_dir = "./raw_data"
processed_img_dir = "./data/images"
processed_mask_dir = "./data/masks"
target_size = (300, 300)

os.makedirs(processed_img_dir, exist_ok=True)
os.makedirs(processed_mask_dir, exist_ok=True)

for root, dirs, files in os.walk(raw_data_dir):
    for file in files:
        file_path = os.path.join(root, file)

        # Skip non-image files
        if not (file.lower().endswith(".png")):
            continue

        if "masks" in root.lower():
            # Move masks into ./data/masks/
            dest_path = os.path.join(processed_mask_dir, file)
            shutil.copy(file_path, dest_path)
        else:
            # Read the image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
            clahe_img = clahe.apply(img)

            dest_path = os.path.join(processed_img_dir, file)
            print(dest_path)
            cv2.imwrite(dest_path, clahe_img)"""

"""cv2.imshow("img", img)
cv2.imshow("CLAHE", clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""