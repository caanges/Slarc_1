import cv2
import math
import os

OUT_DIR = r"H:\Programmering\dva513\Slarc_1\Data\classified_data"

def load_data(pic_id, labeling_id):
    img_path =r"H:\Programmering\dva513\Slarc_1\Data\Data_img\Gen_data"
    image_id = f"img{labeling_id}_{pic_id:04d}.png"
    full_path = os.path.join(img_path, image_id)

    labeling_path = r"H:\Programmering\dva513\Slarc_1\Data\Data_img\Gen_data"
    labeling_id = f"img{labeling_id}_{pic_id:04d}.txt"
    ful_label_path = os.path.join(labeling_path, labeling_id)

    img = cv2.imread(full_path)
    h, w, _ = img.shape
    if img is None:
        print("Failed to load image")
        return

    print("\n________________\n")
    print(labeling_id)
    print(image_id)

    create_data(ful_label_path, image_id, img, h, w)

def denorm(x, y, w, h):
    return int(x * w), int(y * h)

def create_data(ful_label_path, image_id, img, h, w):
    with open(ful_label_path, "r") as f:

        for line in f.readlines():
            data = list(map(float, line.strip().split()))

            class_id = int(data[0])
            cx, cy, bw, bh = data[1:5]

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ---- keypoints ----
            keypoints = data[5:]

            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i:i+3]

                if v == 0:
                    continue  # invisible / ignore

                px, py = denorm(x, y, w, h)

                color = (0, 0, 255) if v == 2 else (255, 0, 0)

                cv2.circle(img, (px, py), 1, color, -1)
            
            out_path = os.path.join(OUT_DIR, image_id)
            save_img(out_path, img)

def save_img(out_path, img):
    cv2.imwrite(out_path, img)

def main():
    number_of_scenes = 5
    size_per_scene = 5
    for j in range(0, number_of_scenes):
        for i in range(0, size_per_scene):
            load_data(i, j)
    print("Done")
        
main()