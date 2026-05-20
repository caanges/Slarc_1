import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "PnP"))
from FinalePnP import PnP_processing

MODEL_PATH = r"C:\Users\een23013\Slarc_1\scripts\yolo\runs\pose\runs\pose\yolov8n_custom_new-5\weights\best.pt"
IMAGE_PATH = r"C:\Users\een23013\Downloads\validation_bridge_occlusion\validation_bridge_occlusion\image5015.png"

model = YOLO(MODEL_PATH)
pnp = PnP_processing()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    cv2.aruco.DetectorParameters()
)

MARKER_SIZE = 0.20
TARGET_IDS = [0,69]

frame = cv2.imread(IMAGE_PATH)
annotated = frame.copy()

# =========================================================
# AXIS FOR VISUALIZATION
# =========================================================

axis = np.float32([
    [0,0,0],
    [0.2,0,0],   # X
    [0,0.2,0],   # Y
    [0,0,0.2]    # Z
]).reshape(-1,3)

# =========================================================
# YOLO PNP
# =========================================================

results = model(frame, conf=0.25, verbose=False)
result = results[0]

forward_yolo_2d = None
forward_yolo_3d = None

yaw_yolo = None

rvec_yolo = None
tvec_yolo = None
R_yolo = None

if result.keypoints is not None and len(result.keypoints.xy) > 0:

    kp = result.keypoints.xy[0].cpu().numpy()
    conf = result.keypoints.conf[0].cpu().numpy()

    valid_2d = []
    valid_3d = []

    # =========================================================
    # PLOT ALL KEYPOINTS
    # =========================================================

    for i in range(len(kp)):

        x, y = kp[i]

        cv2.circle(
            annotated,
            (int(x), int(y)),
            5,
            (0,0,255),
            -1
        )

        cv2.putText(
            annotated,
            f"{i}",
            (int(x)+5, int(y)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            1
        )

    for i in range(min(len(kp), len(pnp.UGV_points_3D))):

        if conf[i] < pnp.conf_threshold:
            continue

        valid_2d.append(kp[i])
        valid_3d.append(pnp.UGV_points_3D[i])

    if len(valid_2d) >= 4:

        obj = np.array(valid_3d, dtype=np.float32).reshape(-1,1,3)
        img = np.array(valid_2d, dtype=np.float32).reshape(-1,1,2)

        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            obj,
            img,
            pnp.camera_matrix,
            pnp.dist_coeffs
        )

        if ok:

            rvec_yolo = rvec
            tvec_yolo = tvec

            R_yolo, _ = cv2.Rodrigues(rvec)

            # =========================================================
            # TEST AXIS
            # =========================================================

            forward = R_yolo[:,1]

            forward /= np.linalg.norm(forward)

            forward_yolo_3d = forward

            yaw_yolo = np.degrees(
                np.arctan2(forward[0], forward[1])
            )

            # =========================================================
            # DRAW FULL AXIS
            # =========================================================

            imgpts, _ = cv2.projectPoints(
                axis,
                rvec,
                tvec,
                pnp.camera_matrix,
                pnp.dist_coeffs
            )

            imgpts = imgpts.reshape(-1,2).astype(int)

            o = tuple(imgpts[0])

            cv2.line(annotated, o, tuple(imgpts[1]), (0,0,255), 3)
            cv2.line(annotated, o, tuple(imgpts[2]), (0,255,0), 3)
            cv2.line(annotated, o, tuple(imgpts[3]), (255,0,0), 3)

             # =========================================================
            # PLOT YAW DIRECTION
            # =========================================================

            center = tuple(imgpts[0])

            arrow_length = 120

            yaw_rad = np.radians(yaw_yolo)

            end_x = int(
                center[0] - arrow_length * np.sin(yaw_rad)
            )

            end_y = int(
                center[1] + arrow_length * np.cos(yaw_rad)
            )

            cv2.arrowedLine(
                annotated,
                center,
                (end_x, end_y),
                (0,255,255),
                4,
                tipLength=0.2
            )

            cv2.putText(
                annotated,
                f"Yaw: {yaw_yolo:.1f}",
                (center[0]+10, center[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )


# =========================================================
# ARUCO
# =========================================================

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

corners, ids, _ = aruco_detector.detectMarkers(gray)

aruco_2d_list = []
aruco_3d_list = []
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

        rvec_gt = rvecs[i]
        tvec_gt = tvecs[i]

        R_gt, _ = cv2.Rodrigues(rvec_gt)

        # =========================================================
        # SAME TEST AXIS
        # =========================================================

        forward = R_gt[:,1]

        forward /= np.linalg.norm(forward)

        aruco_3d_list.append(forward)

        yaw = np.degrees(
            np.arctan2(forward[0], forward[1])
        )

        aruco_yaw_list.append(yaw)

        # =========================================================
        # DRAW FULL AXIS
        # =========================================================

        imgpts, _ = cv2.projectPoints(
            axis,
            rvec_gt,
            tvec_gt,
            pnp.camera_matrix,
            pnp.dist_coeffs
        )

        imgpts = imgpts.reshape(-1,2).astype(int)

        o = tuple(imgpts[0])

        cv2.line(annotated, o, tuple(imgpts[1]), (0,0,255), 3)
        cv2.line(annotated, o, tuple(imgpts[2]), (0,255,0), 3)
        cv2.line(annotated, o, tuple(imgpts[3]), (255,0,0), 3)

        # =========================================================
        # 2D VECTOR
        # =========================================================

        origin = tvec_gt.reshape(3)
        tip = origin + forward * 0.3

        o2, _ = cv2.projectPoints(
            origin.reshape(1,3),
            np.zeros((3,1)),
            np.zeros((3,1)),
            pnp.camera_matrix,
            pnp.dist_coeffs
        )

        t2, _ = cv2.projectPoints(
            tip.reshape(1,3),
            np.zeros((3,1)),
            np.zeros((3,1)),
            pnp.camera_matrix,
            pnp.dist_coeffs
        )

        o2 = o2.ravel()
        t2 = t2.ravel()

        vec2d = np.array([
            t2[0]-o2[0],
            t2[1]-o2[1]
        ])

        vec2d /= np.linalg.norm(vec2d)

        aruco_2d_list.append(vec2d)

# =========================================================
# 2D ERROR
# =========================================================

if (
    forward_yolo_2d is not None and
    len(aruco_2d_list) > 0
):

    aruco2d = np.mean(aruco_2d_list, axis=0)

    aruco2d /= np.linalg.norm(aruco2d)

    dot = np.clip(
        np.dot(forward_yolo_2d, aruco2d),
        -1,
        1
    )

    angle2d = np.degrees(np.arccos(dot))

    print("\n2D ERROR:", angle2d)

# =========================================================
# 3D ERROR
# =========================================================

if (
    forward_yolo_3d is not None and
    len(aruco_3d_list) > 0
):

    aruco3d = np.mean(aruco_3d_list, axis=0)

    aruco3d /= np.linalg.norm(aruco3d)

    dot3 = np.clip(
        np.dot(forward_yolo_3d, aruco3d),
        -1,
        1
    )

    angle3d = np.degrees(np.arccos(dot3))

    print("\n3D ERROR:", angle3d)

# =========================================================
# YAW ERROR
# =========================================================

if (
    yaw_yolo is not None and
    len(aruco_yaw_list) > 0
):

    yaw_aruco = np.mean(aruco_yaw_list)

    yaw_error = abs(yaw_yolo - yaw_aruco)

    yaw_error = min(yaw_error, 360 - yaw_error)

    print("\nYAW ERROR:", yaw_error)

    cv2.putText(
        annotated,
        f"Yaw error: {yaw_error:.2f}",
        (20,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

# =========================================================
# KEYPOINT REMOVAL IMPACT ON YAW
# =========================================================

yaw_aruco = np.mean(aruco_yaw_list)

base_yaw_error = abs(yaw_yolo - yaw_aruco)
base_yaw_error = min(base_yaw_error, 360 - base_yaw_error)

print("\nKEYPOINT REMOVAL IMPACT:")

if (
    yaw_yolo is not None and
    len(aruco_yaw_list) > 0
):

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

        forward_temp = R_temp[:,1]

        forward_temp /= np.linalg.norm(
            forward_temp
        )

        yaw_temp = np.degrees(
            np.arctan2(
                forward_temp[0],
                forward_temp[1]
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
            f"KP {remove_idx} "
            f"removed -> "
            f"Yaw error change: {yaw_change:.3f}"
        )

# =========================================================
# SHOW
# =========================================================

cv2.imshow("DEBUG", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()