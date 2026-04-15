import cv2
import math
import os
import json

def load_images(pic_id, path):
    img_path =r"H:\Programmering\dva513\Slarc_1\Data\Data_img\Gen_data"
    image_id = f"img_{pic_id:04d}.png"
    full_path = os.path.join(img_path, image_id)

    img = cv2.imread(full_path)
    if img is None:
        print("Failed to load image")
        return
   
    load_json_data(pic_id,img)
    #show_images(img)
    save_image(path, img)


def show_images(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_bbox_img(img, bbox, label):
    h, w, _ = img.shape
    x_center, y_center, bw, bh = bbox

    x_center *= w
    y_center *= h
    bw *= w
    bh *= h

    x1 = int(x_center - bw/2)
    y1 = int(y_center - bh/2)
    x2 = int(x_center + bw/2)
    y2 = int(y_center + bh/2)

    if label == 'UGV':
        cv2.rectangle(img, (x1,y1), (x2,y2), (255, 255, 255), 3)
    else:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

def load_json_data(json_id, img):
    global temp_list
    json_path =r"H:\Programmering\dva513\Slarc_1\Data\Data_label\json_data\dataset.json"
   
    json_id_image = (json_id * 5)

    with open(json_path, "r") as file:
        data = json.load(file)

    data_list = list(data.values())

    for i in range(5):
        print(data_list[json_id_image + i])
        bbox = data[f"{json_id_image + i}"]["bbox"]
        label = data[f"{json_id_image + i}"]["object"]
        add_bbox_img(img, bbox, label)

def save_image(path, img):
    cv2.imwrite(path, img)

def main():
    size_of_data = 5
    output_path = r"H:\Programmering\dva513\Slarc_1\Data\classified_data"
    
    for i in range(size_of_data):
        print("\n______________________________________\n")
        img_path = os.path.join(output_path, f"img_{i:04d}.png")
        load_images(i, img_path)

    print("Done")
        
main()