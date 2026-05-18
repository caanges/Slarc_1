import yaml
import os
import shutil

from Coco_yolo import convert_coco_to_yolo
from change_class import fix_classes
from divide_dataset import split_dataset


# ============================================
# LOAD CONFIG
# ============================================

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

base_dir = config["base_dir"]

# ============================================
# AUTO MOVE COCO JSON
# ============================================

train_dir = os.path.join(base_dir, "train")

json_in_train = os.path.join(
    train_dir,
    "_annotations.coco.json"
)

json_in_root = os.path.join(
    base_dir,
    "_annotations.coco.json"
)

# Move JSON automatically if needed
if os.path.exists(json_in_train):

    if not os.path.exists(json_in_root):

        shutil.move(
            json_in_train,
            json_in_root
        )

        print("Moved _annotations.coco.json to testing.coco root")

    else:
        print("JSON already exists in root")

# ============================================
# RUN PIPELINE
# ============================================

convert_coco_to_yolo(config)

fix_classes(config)

split_dataset(config)

print("FULL PIPELINE COMPLETE")