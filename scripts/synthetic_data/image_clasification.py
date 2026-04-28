import cv2
import math
import os
import json

def load_images(pic_id, path, labeling_id):
    img_path =r"H:\Programmering\dva513\Slarc_1\Data\Data_img\Gen_data"
    image_id = f"img{labeling_id}_{pic_id:04d}.png"
    full_path = os.path.join(img_path, image_id)

    img = cv2.imread(full_path)
    if img is None:
        print("Failed to load image")
        return
   
    load_json_data(pic_id,img,labeling_id)
    save_image(path, img)

def add_bbox_img(img, bbox, label):
    #get the height and width of the image and get the center coordinates of the bounding box
    h, w, _ = img.shape
    x_center, y_center, bw, bh = bbox

    x_center *= w
    y_center *= h
    bw *= w
    bh *= h

    #get the two coordinates of two coorners which defines the bounding box
    x1 = int(x_center - bw/2)
    y1 = int(y_center - bh/2)
    x2 = int(x_center + bw/2)
    y2 = int(y_center + bh/2)

    if label.startswith('UGV'):
        cv2.rectangle(img, (x1,y1), (x2,y2), (255, 255, 255), 1)
    else:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

def load_json_data(json_id, img, labeling_id):
    global temp_list
    json_path =r"H:\Programmering\dva513\Slarc_1\Data\Data_label\json_data\dataset.json"
   
    with open(json_path, "r") as file:
        data = json.load(file)

    data_list = list(data.values())
    num_off_attr = 80
    attr_pic = 16
    for i in range(0, attr_pic):
        #get the rigth index when reading the data
        index = labeling_id * num_off_attr + json_id * attr_pic + i
        print(index, data_list[index])
        bbox = data[f"{index}"]["bbox"]
        label = data[f"{index}"]["object"]
        add_bbox_img(img, bbox, label)

def save_image(path, img):
    cv2.imwrite(path, img)

def main():
    size_of_data = 5
    output_path = r"H:\Programmering\dva513\Slarc_1\Data\classified_data"
    for j in range(0, 5):
        for i in range(size_of_data):
            print("\n______________________________________\n")
            img_path = os.path.join(output_path, f"img{j}_{i:04d}.png")
            load_images(i, img_path, j)

    print("Done")
        
main()