import os
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
            cv2.imwrite(dest_path, clahe_img)

            """cv2.imshow("img", img)
            cv2.imshow("CLAHE", clahe_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""