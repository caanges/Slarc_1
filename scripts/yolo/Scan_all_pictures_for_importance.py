import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
import os
import glob

sys.path.append(str(Path(__file__).resolve().parents[1] / "PnP"))
from FinalePnP import PnP_processing

MODEL_PATH = r"C:\Users\een23013\Slarc_1\scripts\yolo\runs\pose\runs\pose\yolov8n_custom_new-5\weights\best.pt"

IMAGE_DIR = r"C:\Users\een23013\Downloads\validation_bridge_occlusion\validation_bridge_occlusion"

model = YOLO(MODEL_PATH)
pnp = PnP_processing()

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_100
)

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    cv2.aruco.DetectorParameters()
)

MARKER_SIZE = 0.20
TARGET_IDS = [0,69]

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

# =========================================================
# GLOBAL YAW IMPORTANCE ACCUMULATOR
# =========================================================

yaw_importance_sum = None
yaw_importance_count = None

# =========================================================
# LOOP THROUGH ALL IMAGES
# =========================================================

for IMAGE_PATH in image_paths:

    print("\n=================================================")
    print("IMAGE:", os.path.basename(IMAGE_PATH))
    print("=================================================")

    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        continue

    results = model(frame, conf=0.25, verbose=False)
    result = results[0]

    yaw_yolo = None

    valid_2d = []
    valid_3d = []

    # =========================================================
    # YOLO PNP
    # =========================================================

    if result.keypoints is not None and len(result.keypoints.xy) > 0:

        kp = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()

        for i in range(min(len(kp), len(pnp.UGV_points_3D))):

            if conf[i] < pnp.conf_threshold:
                continue

            valid_2d.append(kp[i])
            valid_3d.append(pnp.UGV_points_3D[i])

        if len(valid_2d) >= 4:

            obj = np.array(
                valid_3d,
                dtype=np.float32
            ).reshape(-1,1,3)

            img = np.array(
                valid_2d,
                dtype=np.float32
            ).reshape(-1,1,2)

            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj,
                img,
                pnp.camera_matrix,
                pnp.dist_coeffs
            )

            if ok:

                R_yolo, _ = cv2.Rodrigues(rvec)

                forward = R_yolo[:,2]

                forward /= np.linalg.norm(forward)

                yaw_yolo = np.degrees(
                    np.arctan2(
                        forward[0],
                        forward[2]
                    )
                )

    # =========================================================
    # ARUCO
    # =========================================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco_detector.detectMarkers(gray)

    aruco_yaw_list = []

    if ids is not None:

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            MARKER_SIZE,
            pnp.camera_matrix,
            pnp.dist_coeffs
        )

        for i in range(len(ids)):

            if ids[i][0] not in TARGET_IDS:
                continue

            R_gt, _ = cv2.Rodrigues(rvecs[i])

            forward_gt = R_gt[:,2]

            forward_gt /= np.linalg.norm(forward_gt)

            yaw_gt = np.degrees(
                np.arctan2(
                    forward_gt[0],
                    forward_gt[2]
                )
            )

            aruco_yaw_list.append(yaw_gt)

    # =========================================================
    # BASE YAW ERROR
    # =========================================================

    if (
        yaw_yolo is None or
        len(aruco_yaw_list) == 0
    ):
        continue

    yaw_aruco = np.mean(aruco_yaw_list)

    base_yaw_error = abs(
        yaw_yolo - yaw_aruco
    )

    base_yaw_error = min(
        base_yaw_error,
        360 - base_yaw_error
    )

    print("\nBASE YAW ERROR:", base_yaw_error)

    # =========================================================
    # KEYPOINT REMOVAL IMPACT ON YAW ERROR
    # =========================================================

    print("\nKEYPOINT REMOVAL IMPACT:")

    for remove_idx in range(len(valid_2d)):

        temp_2d = []
        temp_3d = []

        for j in range(len(valid_2d)):

            if j == remove_idx:
                continue

            temp_2d.append(valid_2d[j])
            temp_3d.append(valid_3d[j])

        if len(temp_2d) < 4:
            continue

        obj_temp = np.array(
            temp_3d,
            dtype=np.float32
        ).reshape(-1,1,3)

        img_temp = np.array(
            temp_2d,
            dtype=np.float32
        ).reshape(-1,1,2)

        ok_temp, rvec_temp, tvec_temp, _ = cv2.solvePnPRansac(
            obj_temp,
            img_temp,
            pnp.camera_matrix,
            pnp.dist_coeffs
        )

        if not ok_temp:
            continue

        R_temp, _ = cv2.Rodrigues(rvec_temp)

        forward_temp = R_temp[:,2]

        forward_temp /= np.linalg.norm(
            forward_temp
        )

        yaw_temp = np.degrees(
            np.arctan2(
                forward_temp[0],
                forward_temp[2]
            )
        )

        temp_yaw_error = abs(
            yaw_temp - yaw_aruco
        )

        temp_yaw_error = min(
            temp_yaw_error,
            360 - temp_yaw_error
        )

        yaw_change = abs(
            temp_yaw_error - base_yaw_error
        )

        print(
            f"KP {remove_idx} removed -> "
            f"Yaw error change: {yaw_change:.4f}"
        )

        # =========================================================
        # ACCUMULATE GLOBAL IMPORTANCE
        # =========================================================

        if yaw_importance_sum is None:

            yaw_importance_sum = np.zeros(
                len(valid_2d)
            )

            yaw_importance_count = np.zeros(
                len(valid_2d)
            )

        yaw_importance_sum[remove_idx] += yaw_change
        yaw_importance_count[remove_idx] += 1

# =========================================================
# FINAL GLOBAL IMPORTANCE
# =========================================================

print("\n=================================================")
print("FINAL GLOBAL YAW IMPORTANCE")
print("=================================================")

if yaw_importance_sum is None:

    print("No valid data")

else:

    avg_importance = (
        yaw_importance_sum /
        np.maximum(yaw_importance_count, 1)
    )

    ranking = list(
        enumerate(avg_importance)
    )

    ranking.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for kp_idx, score in ranking:

        print(
            f"KP {kp_idx}: "
            f"{score:.4f}"
        )