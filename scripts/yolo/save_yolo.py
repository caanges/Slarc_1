from ultralytics import YOLO
import os

def main():
    # Load trained model
    model = YOLO(r"C:\Users\egn23014\Slarc_1\runs\pose\runs\pose\yolov8n_custom-6\weights\best.pt") #ändra till där modellen finns

    # Input images (unlabeled)
    image_folder = (r"C:\Users\egn23014\Downloads\dataset_2000\split_1") #Ändra till rätt

    # Output labels
    label_folder = (r"C:\Users\egn23014\Downloads\dataset_2000\split_1_yolo_confidence") #Ändra till rätt
    os.makedirs(label_folder, exist_ok=True)

    results = model(image_folder, conf=0.25)

    for r in results:
        if r.boxes is None:
            continue

        img_name = os.path.basename(r.path)
        label_path = os.path.join(label_folder, img_name.replace(".png", ".txt"))

        with open(label_path, "w") as f:
            boxes = r.boxes.xywhn  
            keypoints_xy = r.keypoints.xy
            keypoints_conf = r.keypoints.conf

            for box, kpts_xy, kpts_conf in zip(boxes, keypoints_xy, keypoints_conf):
                class_id = 0  

                line = [class_id] + box.tolist()

                for (x, y), conf in zip(kpts_xy, kpts_conf):
                    line.extend([float(x), float(y), float(conf)])

                f.write(" ".join(map(str, line)) + "\n")


if __name__ == "__main__":
    main()