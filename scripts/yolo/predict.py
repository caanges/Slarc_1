from ultralytics import YOLO
import cv2


def main():
    # Load trained model (after training)
    model = YOLO(r"C:\Users\egn23014\Slarc_1\runs\pose\runs\pose\yolov8n_custom-6\weights\best.pt")

    # Run prediction
    results = model(r"C:\Users\egn23014\Downloads\dataset\dataset\images\val\image078.png", save=True, conf=0.25)

    # Show result with keypoints
    for r in results:
        img = r.plot()  # draws boxes + keypoints
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()