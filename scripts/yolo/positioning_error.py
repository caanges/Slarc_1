"""DGI camera spex
w, h = 1920, 1080 # Ändra om din video har en annan upplösning!
fov_deg = 83.0
fx = w / (2 * math.tan(math.radians(fov_deg) / 2))
fy = fx # Antag kvadratiska pixlar

# Replace these with your real camera calibration values
CAMERA_MATRIX = np.array([
    [fx, 0, w/2],
    [0, fy, h/2],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.array([0.05, -0.05, 0, 0], dtype=np.float32) # En liten gissning på distorsion
"""
import cv2
import numpy as np
import math
import sys
import os
from ultralytics import YOLO

# =========================
# IMPORT PNP FILE
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PNP_DIR = os.path.join(CURRENT_DIR, "..", "PnP")
sys.path.append(PNP_DIR)

from FinalePnP import PnP_processing


# =========================
# SETTINGS
# =========================

IMAGE_PATH = r"C:\Users\egn23014\Downloads\position_validation\position_validation\image6001.png"
MODEL_PATH = r"C:\Users\egn23014\Slarc_1\runs\pose\runs\pose\yolov8n_custom-13\weights\best_real_world.pt"
# YOLO label file used by your PnP code


MARKER_SIZE = 0.18  # meters

focal_length = 640
center = (640 / 2, 400 / 2)

CAMERA_MATRIX = np.array([
    [798.06, 0, 623.26],
    [0, 797.95, 392.96],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.array([24.744, -41.389, -4.426e-05, 0.00094825, 21.19, 24.117, -39.56, 19.846, 0, 0, 0, 0, -0.0011862, 0.005456], dtype=np.float32)

# Offset from UGV center to ArUco tag center, in meters
MARKER_OFFSETS = {
    0: np.array([0.00, 1.00, -0.44], dtype=np.float32),
    69: np.array([1.00, 0.00, -0.44], dtype=np.float32),
}


def rotation_vector_to_matrix(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return rotation_matrix


def detect_aruco_ground_truth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        print("No ArUco markers detected.")
        return image, None, []

    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        MARKER_SIZE,
        CAMERA_MATRIX,
        DIST_COEFFS
    )

    results = []
    ugv_centers = []

    for i, marker_id in enumerate(ids.flatten()):
        rvec = rvecs[i][0]
        tvec = tvecs[i][0]

        tag_center_camera = tvec

        cv2.drawFrameAxes(
            image,
            CAMERA_MATRIX,
            DIST_COEFFS,
            rvec,
            tvec,
            MARKER_SIZE * 0.7
        )

        marker_center_px = corners[i][0].mean(axis=0)
        marker_center_px = tuple(marker_center_px.astype(int))

        cv2.circle(image, marker_center_px, 5, (0, 0, 255), -1)

        cv2.putText(
            image,
            f"ID {marker_id}",
            (marker_center_px[0] + 10, marker_center_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        print("\n==============================")
        print(f"Marker ID: {marker_id}")
        print("==============================")
        print(
            f"Tag center in camera frame: "
            f"x={tag_center_camera[0]:.3f} m, "
            f"y={tag_center_camera[1]:.3f} m, "
            f"z={tag_center_camera[2]:.3f} m"
        )

        if marker_id not in MARKER_OFFSETS:
            print("No offset defined for this marker ID.")
            continue

        offset_ugv_to_marker = MARKER_OFFSETS[marker_id]

        R_marker_to_camera = rotation_vector_to_matrix(rvec)
        offset_camera = R_marker_to_camera @ offset_ugv_to_marker

        ugv_center_camera = tag_center_camera - offset_camera

        ugv_centers.append(ugv_center_camera)

        result = {
            "marker_id": int(marker_id),
            "tag_center_camera": tag_center_camera,
            "ugv_center_camera": ugv_center_camera,
            "rvec": rvec,
            "tvec": tvec,
        }

        results.append(result)

        print(
            f"UGV center estimate from marker {marker_id}: "
            f"x={ugv_center_camera[0]:.3f} m, "
            f"y={ugv_center_camera[1]:.3f} m, "
            f"z={ugv_center_camera[2]:.3f} m"
        )

    if len(ugv_centers) == 0:
        return image, None, results

    final_aruco_center = np.mean(ugv_centers, axis=0)

    print("\n==============================")
    print(f"FINAL ARUCO UGV CENTER from {len(ugv_centers)} marker(s)")
    print("==============================")
    print(
        f"x={final_aruco_center[0]:.3f} m, "
        f"y={final_aruco_center[1]:.3f} m, "
        f"z={final_aruco_center[2]:.3f} m"
    )

    final_center_2d, _ = cv2.projectPoints(
        final_aruco_center.reshape(1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        CAMERA_MATRIX,
        DIST_COEFFS
    )

    cx, cy = final_center_2d[0][0]
    cx = int(cx)
    cy = int(cy)

    cv2.circle(image, (cx, cy), 4, (255, 0, 255), -1)

    cv2.putText(
        image,
        "ArUco center",
        (cx + 10, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2
    )

    return image, final_aruco_center, results


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Could not read image:")
        print(IMAGE_PATH)
        return

    # =========================
    # ARUCO GROUND TRUTH
    # =========================

    output_image, aruco_center_m, aruco_results = detect_aruco_ground_truth(image)

    print("\nNumber of usable ArUco detections:", len(aruco_results))

    # =========================
    # LOAD YOLO MODEL
    # =========================

    model = YOLO(MODEL_PATH)

    # =========================
    # RUN YOLO
    # =========================

    results = model(IMAGE_PATH, conf=0.25, imgsz=640)

    # =========================
    # CREATE PNP PROCESSOR
    # =========================

    processor = PnP_processing()

    pnp_center_m = None

    # =========================
    # RUN PNP FROM YOLO KEYPOINTS
    # =========================

    for r in results:

        if r.keypoints is None:
            print("No keypoints detected.")
            continue

        keypoints_xy = r.keypoints.xy[0].cpu().numpy()
        keypoints_conf = r.keypoints.conf[0].cpu().numpy()

        rvec, tvec = processor.CalculatePoseFromKeypoints(
            keypoints_xy,
            keypoints_conf
        )

        if rvec is not None and tvec is not None:

            # Convert cm -> meters
            pnp_center_m = np.array(tvec).flatten() / 100.0

            print("\n==============================")
            print("PNP CENTER")
            print("==============================")

            print(
                f"x={pnp_center_m[0]:.3f} m, "
                f"y={pnp_center_m[1]:.3f} m, "
                f"z={pnp_center_m[2]:.3f} m"
            )

            # =========================
            # DRAW PNP CENTER
            # =========================

            pnp_center_cm = np.array([[0, 0, 0]], dtype=np.float32)

            pnp_center_2d, _ = cv2.projectPoints(
                pnp_center_cm,
                rvec,
                tvec,
                CAMERA_MATRIX,
                DIST_COEFFS
            )

            px, py = pnp_center_2d[0][0]
            px = int(px)
            py = int(py)

            cv2.circle(output_image, (px, py), 4, (0, 255, 255), -1)

            cv2.putText(
                output_image,
                "PnP center",
                (px + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # =========================
    # COMPARE ARUCO VS PNP
    # =========================

    if aruco_center_m is not None and pnp_center_m is not None:

        error_m = np.linalg.norm(
            pnp_center_m - aruco_center_m
        )

        print("\n==============================")
        print("POSITION ERROR")
        print("==============================")

        print(
            f"ArUco center: "
            f"x={aruco_center_m[0]:.3f}, "
            f"y={aruco_center_m[1]:.3f}, "
            f"z={aruco_center_m[2]:.3f}"
        )

        print(
            f"PnP center:   "
            f"x={pnp_center_m[0]:.3f}, "
            f"y={pnp_center_m[1]:.3f}, "
            f"z={pnp_center_m[2]:.3f}"
        )

        print(f"\n3D position error: {error_m:.3f} m")
        print(f"3D position error: {error_m * 100:.1f} cm")

    # =========================
    # SHOW IMAGE
    # =========================

    screen_width = 1280
    screen_height = 720

    h, w = output_image.shape[:2]
    scale = min(screen_width / w, screen_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(output_image, (new_w, new_h))

    cv2.imshow("ArUco ground truth + PnP error", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()