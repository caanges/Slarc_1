import cv2
import math
import os

OUT_DIR = r"C:\Data_dva513\Data\classified_data"

def load_data(pic_id, labeling_id):
    img_path =r"C:\Data_dva513\Data\Train_val_test\images\val"
    image_id = f"img{labeling_id}_{pic_id:04d}.png"
    full_path = os.path.join(img_path, image_id)

    labeling_path = r"C:\Data_dva513\Data\Train_val_test\labels\val"
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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # ---- keypoints ----
            keypoints = data[5:]

            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i:i+3]

                px, py = denorm(x, y, w, h)

                if v == 2:
                    color = (0, 255, 0)  
                elif v == 1:
                    color = (255, 0, 0)
                elif v == 0:
                    color = (0, 0, 255)

                cv2.circle(img, (px, py), 1, color, -1)
            
            out_path = os.path.join(OUT_DIR, image_id)
            save_img(out_path, img)

def save_img(out_path, img):
    cv2.imwrite(out_path, img)

def main():
    start_scene = 7 #0-5
    start_scene_levels = start_scene * 5
    number_of_scenes = 8 #1-6
    number_of_levels = number_of_scenes * 5
    size_per_scene = 50
    for j in range(start_scene_levels, number_of_levels):
        for i in range(0, size_per_scene):
            load_data(i, j)
    print("Done")
        
main()