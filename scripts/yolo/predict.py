from ultralytics import YOLO
import cv2


def main():
    # Load trained model (after training)
    model = YOLO(r"C:\Users\egn23014\Slarc_1\runs\pose\runs\pose\yolov8n_custom-9\weights\best.pt")#ändra till där modellen finns

    # Run prediction

    results = model(r"C:\Users\egn23014\Downloads\high_picture_test\high_picture_test\image017.png", save=True, conf=0.25)

    # Show result with keypoints
    for r in results:
        img = r.plot()  # original image with boxes + keypoints

        # Resize for display
        screen_width = 1280
        screen_height = 720

        h, w = img.shape[:2]
        scale = min(screen_width / w, screen_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(img, (new_w, new_h))

        cv2.imshow("Prediction", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()