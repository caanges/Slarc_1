# This script splits the dataset into training and validation sets, and organizes them into the expected directory structure for YOLOv8.

import os
import random
import shutil

# Paths
images_dir = "images"   # You need to have the actual image files in this folder
labels_dir = "labels"   # You need to have the corresponding label files in this folder
output_dir = "dataset" 

# Split ratio, e.g. 0.9 means 90% training, 10% validation
train_ratio = 0.9

# Get all image files
images = [f for f in os.listdir(images_dir) if f.endswith(".png")]

# Shuffle randomly
random.shuffle(images)

# Split
split_index = int(len(images) * train_ratio)
train_files = images[:split_index]
val_files = images[split_index:]

def copy_files(file_list, subset):
    for file in file_list:
        base_name = os.path.splitext(file)[0]

        img_src = os.path.join(images_dir, file)
        lbl_src = os.path.join(labels_dir, base_name + ".txt")

        img_dst = os.path.join(output_dir, "images", subset, file)
        lbl_dst = os.path.join(output_dir, "labels", subset, base_name + ".txt")

        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

        shutil.copy(img_src, img_dst)

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)
        else:
            print(f"Warning: Missing label for {file}")

# Copy files
copy_files(train_files, "train")    # You can change "train" if you want a different structure
copy_files(val_files, "val")        # You can change "val" if you want a different structure

print("Done!")