from ultralytics import YOLO


def main():
    # Load base YOLOv8 nano pose model
    model = YOLO(r"H:\Programmering\dva513\Slarc_1\scripts\yolo\runs\pose\runs\pose\yolov8n_custom-6\weights\best.pt")

    # Train
    model.train(
        data=r"H:\Programmering\dva513\Slarc_1\scripts\yolo\data.yaml",  # path to your dataset config
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",           # use "cpu" if no GPU
        workers=4,
        project="runs/pose",
        name="yolov8n_custom"
    )


if __name__ == "__main__":
    main()