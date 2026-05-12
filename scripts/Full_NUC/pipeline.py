import cv2
import depthai as dai
from ultralytics import YOLO
import time
import sys
from pathlib import Path
import numpy as np

# Import PnP file from ../PnP/FinalePnP.py
sys.path.append(str(Path(__file__).resolve().parents[1] / "PnP"))
from FinalePnP import PnP_processing

MODEL_PATH = r"C:\Users\egn23014\Slarc_1\runs\pose\runs\pose\yolov8n_custom-9\weights\best.pt"

model = YOLO(MODEL_PATH)
pnp = PnP_processing()

# Axis to draw on the UGV
axis = np.float32([
    [0, 0, 0],
    [15, 0, 0],
    [0, 15, 0],
    [0, 0, 15]
])

device = dai.Device()

with dai.Pipeline(device) as pipeline:

    cam = pipeline.create(dai.node.Camera).build()
    q = cam.requestOutput(size=(640, 480)).createOutputQueue()

    pipeline.start()

    prev_time = time.time()

    print("Press Q to quit.")

    while True:
        frame_msg = q.get()
        frame = frame_msg.getCvFrame()

        results = model(frame, conf=0.25, verbose=False)
        result = results[0]

        annotated = result.plot()

        if result.keypoints is not None and len(result.keypoints.xy) > 0:

            keypoints_xy = result.keypoints.xy[0].cpu().numpy()
            keypoints_conf = result.keypoints.conf[0].cpu().numpy()

            valid_2d_points = []
            matching_3d_points = []

            for i in range(min(len(keypoints_xy), len(pnp.UGV_points_3D))):
                x, y = keypoints_xy[i]
                conf = keypoints_conf[i]

                if conf >= pnp.conf_threshold:
                    valid_2d_points.append([x, y])
                    matching_3d_points.append(pnp.UGV_points_3D[i])

            if len(valid_2d_points) >= 4:
                object_points = np.array(matching_3d_points, dtype=np.float32).reshape(-1, 1, 3)
                image_points = np.array(valid_2d_points, dtype=np.float32).reshape(-1, 1, 2)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    pnp.camera_matrix,
                    pnp.dist_coeffs,
                    iterationsCount=100,
                    reprojectionError=20.0,
                    flags=cv2.SOLVEPNP_EPNP
                )

                if success:
                    r = rvec.flatten()
                    t = tvec.flatten()

                    # Project 3D axis into image
                    imgpts, _ = cv2.projectPoints(
                        axis,
                        rvec,
                        tvec,
                        pnp.camera_matrix,
                        pnp.dist_coeffs
                    )

                    imgpts = imgpts.astype(int)
                    o, x_axis, y_axis, z_axis = imgpts.reshape(-1, 2)

                    # Draw axis
                    cv2.line(annotated, tuple(o), tuple(x_axis), (0, 0, 255), 3)    # X red
                    cv2.line(annotated, tuple(o), tuple(y_axis), (0, 255, 0), 3)    # Y green
                    cv2.line(annotated, tuple(o), tuple(z_axis), (255, 0, 0), 3)    # Z blue

                    cv2.putText(
                        annotated,
                        f"tvec X:{t[0]:.1f} Y:{t[1]:.1f} Z:{t[2]:.1f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"rvec X:{r[0]:.2f} Y:{r[1]:.2f} Z:{r[2]:.2f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

                    print("rvec:", r)
                    print("tvec:", t)

                else:
                    cv2.putText(
                        annotated,
                        "PnP failed",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            else:
                cv2.putText(
                    annotated,
                    f"Not enough PnP points: {len(valid_2d_points)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLOv8 Pose + PnP Axis", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()