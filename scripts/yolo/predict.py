from ultralytics import YOLO
import cv2


def main():
    # Load trained model (after training)
    model = YOLO("runs/pose/yolov8n_custom/weights/best.pt")

    # Run prediction
    results = model("test_image.jpg", save=True, conf=0.25)

    # Show result with keypoints
    for r in results:
        img = r.plot()  # draws boxes + keypoints
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()