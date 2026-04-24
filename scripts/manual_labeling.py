import cv2
import os
import json

#ändra till path du har till bilden
path = 
dataset{}

def click_event(event, x, y, flag, param):
    data_id = len(dataset)
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 2, (0,0,255), -1)
        dataset[data_id] = f"{(x, y)}\n"

def make_txt():

    

def open_image(id):
    image_path = os.path.join(path, f"image{id:03d}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error open image")
        return
    cv2.imshow(f"image{id:03d}", image)
    cv2.setMouseCallback(f"image{id:03d}", click_event)

    cv2.waitkey(0)
    cv2.destroyAllWindows()



def main():
    data_size = 2
    for i in range(data_size):
        open_image(i)

if __name__ == "__main__":
    main()