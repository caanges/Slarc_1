# pip install opencv-contrib-python

import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

# ============================================
# IMPORT PNP
# ============================================

sys.path.append(str(Path(__file__).resolve().parents[1] / "PnP"))
from FinalePnP import PnP_processing

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = r"C:\Users\een23013\Slarc_1\scripts\yolo\runs\pose\runs\pose\yolov8n_custom_new-5\weights\best.pt"

IMAGE_PATH = r"C:\Users\een23013\Downloads\image3001.png"

model = YOLO(MODEL_PATH)

pnp = PnP_processing()

# ============================================
# ARUCO SETTINGS
# ============================================

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_100
)

aruco_params = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    aruco_params
)

# ============================================
# REAL MARKER SIZE IN METERS
# ============================================

MARKER_SIZE = 0.20

# ============================================
# TARGET IDS
# ============================================

TARGET_IDS = [0, 69]

# ============================================
# AXIS FOR DRAWING
# ============================================

axis = np.float32([
    [0, 0, 0],
    [-15, 0, 0],   # X axis
    [0, -15, 0],   # Y axis
    [0, 0, 15]     # Z axis
])

# ============================================
# LOAD IMAGE
# ============================================

frame = cv2.imread(IMAGE_PATH)

if frame is None:

    print("Could not load image")

    exit()

# ============================================
# RUN YOLO
# ============================================

results = model(frame, conf=0.25, verbose=False)

result = results[0]

annotated = result.plot()

# ============================================
# YOLO PNP
# ============================================

forward_yolo = None

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

            matching_3d_points.append(
                pnp.UGV_points_3D[i]
            )

    if len(valid_2d_points) >= 4:

        object_points = np.array(
            matching_3d_points,
            dtype=np.float32
        ).reshape(-1, 1, 3)

        image_points = np.array(
            valid_2d_points,
            dtype=np.float32
        ).reshape(-1, 1, 2)

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

            # ============================================
            # ROTATION MATRIX
            # ============================================

            R_yolo, _ = cv2.Rodrigues(rvec)

            # ============================================
            # FORWARD VECTOR
            # ============================================

            # You may later need:
            # R[:,0]
            # R[:,1]
            # -R[:,2]

            forward_yolo = R_yolo[:, 2]

            forward_yolo = (
                forward_yolo /
                np.linalg.norm(forward_yolo)
            )

            print("\nYOLO Forward:")
            print(forward_yolo)

            # ============================================
            # DRAW YOLO AXIS
            # ============================================

            imgpts, _ = cv2.projectPoints(

                axis,

                rvec,

                tvec,

                pnp.camera_matrix,

                pnp.dist_coeffs
            )

            imgpts = imgpts.astype(int)

            o, x_axis, y_axis, z_axis = imgpts.reshape(-1, 2)

            cv2.line(
                annotated,
                tuple(o),
                tuple(x_axis),
                (0, 0, 255),
                3
            )

            cv2.line(
                annotated,
                tuple(o),
                tuple(y_axis),
                (0, 255, 0),
                3
            )

            cv2.line(
                annotated,
                tuple(o),
                tuple(z_axis),
                (255, 0, 0),
                3
            )

# ============================================
# ARUCO DETECTION
# ============================================

gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)

corners, ids, rejected = aruco_detector.detectMarkers(
    gray
)

print("\nDetected IDs:", ids)

print("Rejected candidates:", len(rejected))

# ============================================
# STORE ARUCO FORWARD VECTORS
# ============================================

forward_vectors = []

if ids is not None:

    # ============================================
    # DRAW DETECTED MARKERS
    # ============================================

    cv2.aruco.drawDetectedMarkers(

        annotated,

        corners,

        ids
    )

    # ============================================
    # POSE ESTIMATION
    # ============================================

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(

        corners,

        MARKER_SIZE,

        pnp.camera_matrix,

        pnp.dist_coeffs
    )

    # ============================================
    # LOOP THROUGH DETECTED TAGS
    # ============================================

    for i in range(len(ids)):

        marker_id = ids[i][0]

        print(f"\nMarker ID: {marker_id}")

        # ============================================
        # IGNORE OTHER IDS
        # ============================================

        if marker_id not in TARGET_IDS:

            continue

        print("Using this marker")

        rvec_gt = rvecs[i]
        tvec_gt = tvecs[i]

        # ============================================
        # DRAW AXIS
        # ============================================

        cv2.drawFrameAxes(

            annotated,

            pnp.camera_matrix,

            pnp.dist_coeffs,

            rvec_gt,

            tvec_gt,

            0.10
        )

        # ============================================
        # ROTATION MATRIX
        # ============================================

        R_gt, _ = cv2.Rodrigues(rvec_gt)

        # ============================================
        # FORWARD VECTOR
        # ============================================

        # You may later need:
        # R_gt[:,0]
        # R_gt[:,1]
        # -R_gt[:,2]

        forward_aruco = R_gt[:, 2]

        forward_aruco = (
            forward_aruco /
            np.linalg.norm(forward_aruco)
        )

        print("\nARUCO Forward:")
        print(forward_aruco)

        # ============================================
        # STORE VECTOR
        # ============================================

        forward_vectors.append(
            forward_aruco
        )

else:

    print("\nNO ARUCO DETECTED")

# ============================================
# ANGULAR ERROR
# ============================================

if (
    forward_yolo is not None and
    len(forward_vectors) > 0
):

    print(
        f"\nUsing {len(forward_vectors)} "
        f"ARUCO marker(s)"
    )

    if len(forward_vectors) == 1:

        print(
            "WARNING: Only one ARUCO "
            "marker detected"
        )

    # ============================================
    # AVERAGE ARUCO FORWARD VECTOR
    # ============================================

    avg_forward = np.mean(
        forward_vectors,
        axis=0
    )

    avg_forward = (
        avg_forward /
        np.linalg.norm(avg_forward)
    )

    print("\nAVERAGE ARUCO FORWARD:")
    print(avg_forward)

    print("\nYOLO FORWARD:")
    print(forward_yolo)

    # ============================================
    # ANGULAR ERROR
    # ============================================

    dot = np.dot(
        forward_yolo,
        avg_forward
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0
    )

    angle_error = np.degrees(
        np.arccos(dot)
    )

    print(
        f"\nAngular Error: "
        f"{angle_error:.2f} deg"
    )

    cv2.putText(

        annotated,

        f"Angular Error: {angle_error:.2f} deg",

        (20, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (0, 255, 255),

        2
    )

else:

    print(
        "\nCould not compute angular error"
    )

# ============================================
# RESIZE FOR DISPLAY
# ============================================

screen_width = 1280
screen_height = 720

h, w = annotated.shape[:2]

scale = min(
    screen_width / w,
    screen_height / h
)

new_w = int(w * scale)

new_h = int(h * scale)

resized_img = cv2.resize(
    annotated,
    (new_w, new_h)
)

# ============================================
# SHOW
# ============================================

cv2.imshow(
    "YOLO vs ARUCO",
    resized_img
)

print("\nPress Q or ESC to quit")

while True:

    key = cv2.waitKey(1)

    # ESC
    if key == 27:
        break

    # Q
    if key == ord("q"):
        break

cv2.destroyAllWindows()

cv2.waitKey(1)