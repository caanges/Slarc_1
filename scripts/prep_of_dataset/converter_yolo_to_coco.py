#convert so that it is correct for Roboflow coco file
import json
from pathlib import Path
from PIL import Image

# ===== CONFIG =====
images_dir = Path(r"C:\Users\egn23014\Downloads\dataset_2000\split_1")
labels_dir = Path(r"C:\Users\egn23014\Downloads\dataset_2000\split_1_labels")
output_json = Path(r"C:\Users\egn23014\Downloads\dataset_2000\split_1_annotations.coco.json")

NUM_KEYPOINTS = 13
CATEGORY_ID = 1

# ==================

coco = {
    "info": {
        "year": "2026",
        "version": "dataset",
        "description": "Converted from YOLO pose",
        "contributor": "",
        "url": "",
        "date_created": ""
    },
    "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
    "categories": [
        {"id": 0, "name": "testing", "supercategory": "none"},
        {"id": 1, "name": "My-First-Project", "supercategory": "testing"}
    ],
    "images": [],
    "annotations": []
}

image_id = 0
annotation_id = 1

image_files = sorted(
    list(images_dir.glob("*.png")) +
    list(images_dir.glob("*.jpg")) +
    list(images_dir.glob("*.jpeg"))
)

for image_path in image_files:
    label_path = labels_dir / f"{image_path.stem}.txt"

    with Image.open(image_path) as img:
        width, height = img.size

    coco["images"].append({
        "id": image_id,
        "license": 1,
        "file_name": image_path.name,
        "height": height,
        "width": width,
        "date_captured": "",
        "extra": {"name": image_path.name}
    })

    if not label_path.exists():
        print(f"⚠️ Missing label for {image_path.name}")
        image_id += 1
        continue

    lines = label_path.read_text().strip().splitlines()

    for line_num, line in enumerate(lines, start=1):
        parts = list(map(float, line.split()))

        expected_len = 5 + NUM_KEYPOINTS * 3
        if len(parts) != expected_len:
            print(f"❌ Skipping {label_path.name} line {line_num}: {len(parts)} values, expected {expected_len}")
            continue

        # YOLO bbox is normalized
        cls = int(parts[0])
        x_center, y_center, w_norm, h_norm = parts[1:5]

        x = (x_center - w_norm / 2) * width
        y = (y_center - h_norm / 2) * height
        w = w_norm * width
        h = h_norm * height

        # Keypoints are already pixel coordinates
        kp = parts[5:]
        keypoints = []
        visible_count = 0

        for j in range(NUM_KEYPOINTS):
            kx = kp[j * 3]
            ky = kp[j * 3 + 1]
            v = int(kp[j * 3 + 2])

            if v > 0:
                visible_count += 1

            keypoints.extend([
                round(kx, 3),
                round(ky, 3),
                v
            ])

        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": CATEGORY_ID,
            "bbox": [
                round(x),
                round(y),
                round(w, 3),
                round(h, 3)
            ],
            "area": round(w * h, 3),
            "segmentation": [],
            "iscrowd": 0,
            "keypoints": keypoints,
            "num_keypoints": visible_count
        })

        annotation_id += 1

    image_id += 1

with open(output_json, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ Saved COCO file to: {output_json}")