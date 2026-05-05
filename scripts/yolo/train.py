from ultralytics import YOLO


def main():
    # Load base YOLOv8 nano pose model
    model = YOLO("yolov8n-pose.pt")

    # Train
    model.train(
        data="data.yaml",  # path to your dataset config
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