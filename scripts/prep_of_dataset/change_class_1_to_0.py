import os

# Folder containing your YOLO label .txt files
LABELS_DIR = r"C:\Users\egn23014\Downloads\testing.coco (1)\Divided_dataset\labels\train"

# Go through every txt file
for filename in os.listdir(LABELS_DIR):
    if filename.endswith(".txt"):

        file_path = os.path.join(LABELS_DIR, filename)

        # Read all lines
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()

            # Replace first number (class id) with 0
            if len(parts) > 0:
                parts[0] = "0"

            new_lines.append(" ".join(parts) + "\n")

        # Write updated lines back
        with open(file_path, "w") as f:
            f.writelines(new_lines)

        print(f"Updated: {filename}")

print("Done! All classes changed to 0.")