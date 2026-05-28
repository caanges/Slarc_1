import os
import random
import shutil


def split_dataset(config):

    base_dir = config["base_dir"]

    images_dir = os.path.join(
        base_dir,
        config["images_dir"]
    )

    labels_dir = os.path.join(
        base_dir,
        config["labels_dir"]
    )

    output_dir = os.path.join(
        base_dir,
        config["dataset_dir"]
    )

    train_ratio = config["train_ratio"]

    # ============================================
    # Create dataset folders automatically
    # ============================================

    os.makedirs(
        os.path.join(
            output_dir,
            "images",
            "train"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            output_dir,
            "images",
            "val"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            output_dir,
            "labels",
            "train"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            output_dir,
            "labels",
            "val"
        ),
        exist_ok=True
    )

    # ============================================
    # Find images
    # ============================================

    images = []

    for f in os.listdir(images_dir):

        if (
            f.endswith(".png")
            or f.endswith(".jpg")
            or f.endswith(".jpeg")
        ):

            images.append(f)

    # Shuffle
    random.shuffle(images)

    # Split
    split_index = int(
        len(images) * train_ratio
    )

    train_files = images[:split_index]
    val_files = images[split_index:]

    # ============================================
    # Copy files
    # ============================================

    def copy_files(file_list, subset):

        for file in file_list:

            base_name = os.path.splitext(file)[0]

            img_src = os.path.join(
                images_dir,
                file
            )

            lbl_src = os.path.join(
                labels_dir,
                base_name + ".txt"
            )

            img_dst = os.path.join(
                output_dir,
                "images",
                subset,
                file
            )

            lbl_dst = os.path.join(
                output_dir,
                "labels",
                subset,
                base_name + ".txt"
            )

            shutil.copy(
                img_src,
                img_dst
            )

            if os.path.exists(lbl_src):

                shutil.copy(
                    lbl_src,
                    lbl_dst
                )

            else:
                print(
                    f"Missing label for {file}"
                )

    # Copy train files
    copy_files(
        train_files,
        "train"
    )

    # Copy validation files
    copy_files(
        val_files,
        "val"
    )

    print("Dataset split complete.")