import os


def fix_classes(config):

    LABELS_DIR = os.path.join(
        config["base_dir"],
        config["labels_dir"]
    )

    if not os.path.exists(LABELS_DIR):

        print(
            f"Labels folder not found: {LABELS_DIR}"
        )

        return

    for filename in os.listdir(LABELS_DIR):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                LABELS_DIR,
                filename
            )

            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []

            for line in lines:

                parts = line.strip().split()

                if len(parts) > 0:
                    parts[0] = "0"

                new_lines.append(
                    " ".join(parts) + "\n"
                )

            with open(file_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated: {filename}")

    print("All classes changed to 0.")