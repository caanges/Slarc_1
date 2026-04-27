import cv2
import depthai as dai
from pathlib import Path

SAVE_DIR = Path(r"C:\Users\edvin\OneDrive\Skrivbord") #change to where you want the pictures
SAVE_DIR.mkdir(exist_ok=True)

print("Press 'q' to quit")
print("Press ENTER to save image.")

with dai.Pipeline() as pipeline:
    # Camera
    camera = pipeline.create(dai.node.Camera).build()
    cam_output = camera.requestOutput((640, 640), dai.ImgFrame.Type.NV12)

    # Convert to grayscale for feature tracking
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    cam_output.link(manip.inputImage)

    # Feature tracker
    feature_tracker = pipeline.create(dai.node.FeatureTracker)
    manip.out.link(feature_tracker.inputImage)

    feature_tracker.initialConfig.setCornerDetector(
        dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
    )
    feature_tracker.initialConfig.setNumTargetFeatures(256)

    corner_detector = dai.FeatureTrackerConfig.CornerDetector()
    corner_detector.numMaxFeatures = 256
    corner_detector.numTargetFeatures = 256

    thresholds = dai.FeatureTrackerConfig.CornerDetector.Thresholds()
    thresholds.initialValue = 20000
    corner_detector.thresholds = thresholds

    feature_tracker.initialConfig.setCornerDetector(corner_detector)

    # RVC2-specific tuning
    feature_tracker.setHardwareResources(2, 2)

    # Host queues
    image_queue = cam_output.createOutputQueue()
    feature_queue = feature_tracker.outputFeatures.createOutputQueue()

    pipeline.start()

    image_number = 1

    while pipeline.isRunning():
        frame = image_queue.get().getCvFrame()
        tracked_features = feature_queue.get().trackedFeatures

        # Draw only the current feature points
        for feature in tracked_features:
            x = int(feature.position.x)
            y = int(feature.position.y)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        cv2.imshow("Feature Points", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            filename = SAVE_DIR / f"image{image_number:03d}.png"
            cv2.imwrite(str(filename), frame)
            print(f"Saved {filename}")
            image_number += 1

        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()