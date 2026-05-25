import json
import os


def convert_coco_to_yolo(config):

    base_dir = config["base_dir"]

    INPUT_JSON = os.path.join(
        base_dir,
        config["input_json"]
    )

    OUTPUT_DIR = os.path.join(
        base_dir,
        config["labels_dir"]
    )

    IMAGES_DIR = os.path.join(
        base_dir,
        config["images_dir"]
    )

    # Create Labels folder automatically
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_JSON) as f:
        data = json.load(f)

    # image_id → image info
    images = {
        img["id"]: img
        for img in data["images"]
    }

    # Group annotations per image
    anns_per_image = {}

    for ann in data["annotations"]:

        anns_per_image.setdefault(
            ann["image_id"],
            []
        ).append(ann)

    for img_id, anns in anns_per_image.items():

        img_info = images[img_id]

        img_w = img_info["width"]
        img_h = img_info["height"]

        original_name = img_info["file_name"]

        name_no_ext, ext = os.path.splitext(original_name)

        # Clean filename
        name_no_ext = name_no_ext.split(".rf.")[0]
        name_no_ext = name_no_ext.replace("_png", "")
        name_no_ext = name_no_ext.replace("_jpg", "")

        short_name = name_no_ext

        # Find actual image file
        actual_file = None

        for f in os.listdir(IMAGES_DIR):

            file_no_ext, file_ext = os.path.splitext(f)

            clean_file_no_ext = file_no_ext.split(".rf.")[0]
            clean_file_no_ext = clean_file_no_ext.replace("_png", "")
            clean_file_no_ext = clean_file_no_ext.replace("_jpg", "")

            if clean_file_no_ext == short_name:
                actual_file = f
                break

        if actual_file is None:
            print(f"Cannot find file for {short_name}")
            continue

        old_img_path = os.path.join(
            IMAGES_DIR,
            actual_file
        )

        ext = os.path.splitext(actual_file)[1]

        new_img_name = short_name + ext

        new_img_path = os.path.join(
            IMAGES_DIR,
            new_img_name
        )

        # Rename image
        if actual_file != new_img_name:

            if not os.path.exists(new_img_path):

                os.rename(
                    old_img_path,
                    new_img_path
                )

                print(f"{actual_file} → {new_img_name}")

            else:
                print(f"Already exists: {new_img_name}")

        # Create YOLO txt label
        txt_path = os.path.join(
            OUTPUT_DIR,
            short_name + ".txt"
        )

        with open(txt_path, "w") as f:

            for ann in anns:

                cls = 0

                # bbox
                x, y, w, h = map(
                    float,
                    ann["bbox"]
                )

                x_center = (
                    x + w / 2
                ) / img_w

                y_center = (
                    y + h / 2
                ) / img_h

                w /= img_w
                h /= img_h

                # keypoints
                kpts = ann["keypoints"]

                kpts_out = []

                for i in range(0, len(kpts), 3):

                    kx = float(kpts[i]) / img_w
                    ky = float(kpts[i + 1]) / img_h
                    v = int(kpts[i + 2])

                    kpts_out.extend([
                        kx,
                        ky,
                        v
                    ])

                line = [
                    cls,
                    x_center,
                    y_center,
                    w,
                    h
                ] + kpts_out

                f.write(
                    " ".join(map(str, line)) + "\n"
                )

    print("COCO → YOLO conversion complete.")