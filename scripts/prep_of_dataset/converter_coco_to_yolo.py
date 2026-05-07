import json
import os

INPUT_JSON = r"C:\Users\egn23014\Downloads\testing.coco (1)\_annotations.coco.json"
OUTPUT_DIR = r"C:\Users\egn23014\Downloads\testing.coco (1)\Labels"
IMAGES_DIR = r"C:\Users\egn23014\Downloads\testing.coco (1)\Images"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_name(filename):
    name, ext = os.path.splitext(filename)

    # Remove Roboflow part: .rf.randomletters
    name = name.split(".rf.")[0]

    # Remove endings like _png, _png_png, _jpg
    while name.endswith("_png") or name.endswith("_jpg"):
        name = name.rsplit("_", 1)[0]

    return name


with open(INPUT_JSON) as f:
    data = json.load(f)

# Map image_id → image info
images = {img["id"]: img for img in data["images"]}

# Group annotations per image
anns_per_image = {}
for ann in data["annotations"]:
    anns_per_image.setdefault(ann["image_id"], []).append(ann)

# List image files once
image_files = os.listdir(IMAGES_DIR)

for img_id, anns in anns_per_image.items():
    img_info = images[img_id]
    img_w, img_h = img_info["width"], img_info["height"]

    original_name = img_info["file_name"]
    short_name = clean_name(original_name)

    # ---- FIND EXACT MATCH AFTER CLEANING ----
    actual_file = None

    for file in image_files:
        if clean_name(file) == short_name:
            actual_file = file
            break

    if actual_file is None:
        print(f"Cannot find file for {short_name}")
        continue

    old_img_path = os.path.join(IMAGES_DIR, actual_file)

    # Keep real extension from disk file
    ext = os.path.splitext(actual_file)[1]
    new_img_name = short_name + ext
    new_img_path = os.path.join(IMAGES_DIR, new_img_name)

    # ---- SAFE RENAME ----
    if actual_file != new_img_name:
        if not os.path.exists(new_img_path):
            os.rename(old_img_path, new_img_path)
            print(f"{actual_file} → {new_img_name}")

            # Update list so script does not search old name again
            image_files.remove(actual_file)
            image_files.append(new_img_name)
        else:
            print(f"Already exists: {new_img_name}")

    # ---- LABEL FILE ----
    txt_path = os.path.join(OUTPUT_DIR, short_name + ".txt")

    with open(txt_path, "w") as f:
        for ann in anns:
            cls = ann["category_id"]

            x, y, w, h = map(float, ann["bbox"])

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w /= img_w
            h /= img_h

            kpts = ann.get("keypoints", [])
            kpts_out = []

            for i in range(0, len(kpts), 3):
                kx = float(kpts[i]) / img_w
                ky = float(kpts[i + 1]) / img_h
                v = int(kpts[i + 2])
                kpts_out.extend([kx, ky, v])

            line = [cls, x_center, y_center, w, h] + kpts_out
            f.write(" ".join(map(str, line)) + "\n")