import cv2
import json
import numpy as np
import os
import sys
from ultralytics import YOLO


# =========================
# IMPORT PNP FILE
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PNP_DIR = os.path.join(CURRENT_DIR, "..", "PnP")
sys.path.append(PNP_DIR)

from PnP_algot_DJI import PnP_processing_algot_DJI


# =========================
# SETTINGS
# =========================

IMAGE_FOLDER = r"C:\Users\egn23014\Pictures\Drone_pics_2"
COCO_JSON = r"C:\Users\egn23014\Documents\_annotations.coco.json"
HEIGHT_TXT = r"C:\Users\egn23014\Pictures\Drone_pics_2\New Textdokument.txt"

MODEL_PATH = r"C:\Users\egn23014\Slarc_1\runs\pose\runs\pose\yolov8n_custom-13\weights\best_real_world.pt"

CONF_THRES = 0.25


# =========================
# READ HEIGHT FILE
# =========================
# Expected format:
# image5001.png 6.35
# image5002.png 6.42

def load_heights(height_txt):
    heights = {}

    with open(height_txt, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            image_name = parts[0]
            height_m = float(parts[1])

            heights[image_name] = height_m

    return heights


# =========================
# READ COCO CENTER POINTS
# =========================
# Assumes your manual center point is stored as keypoints in COCO.
# Uses the first visible keypoint as the UGV center.

def load_coco_centers(coco_json):
    with open(coco_json, "r") as f:
        data = json.load(f)

    images = {
        img["id"]: img
        for img in data["images"]
    }

    centers = {}

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        image_name = images[image_id]["file_name"]

        if "keypoints" not in ann:
            continue

        kpts = ann["keypoints"]

        if len(kpts) < 3:
            continue

        # First keypoint = manual UGV center
        x = float(kpts[0])
        y = float(kpts[1])
        v = int(kpts[2])

        if v == 0:
            continue

        centers[image_name] = np.array([x, y], dtype=np.float32)

    return centers


# =========================
# PROJECT PNP CENTER TO IMAGE
# =========================

def project_pnp_center_to_image(rvec, tvec, camera_matrix, dist_coeffs):
    center_3d = np.array([[0, 0, 0]], dtype=np.float32)

    center_2d, _ = cv2.projectPoints(
        center_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )

    return center_2d[0][0]


# =========================
# MAIN
# =========================

def main():
    model = YOLO(MODEL_PATH)
    processor = PnP_processing_algot_DJI()

    coco_centers = load_coco_centers(COCO_JSON)
    heights = load_heights(HEIGHT_TXT)

    results_summary = []

    for image_name, gt_center_2d in coco_centers.items():
        image_path = os.path.join(IMAGE_FOLDER, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        if image_name not in heights:
            print(f"No height found for: {image_name}")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        yolo_results = model(image_path, conf=CONF_THRES, imgsz=640)

        if (
            len(yolo_results) == 0
            or yolo_results[0].keypoints is None
            or yolo_results[0].keypoints.xy is None
            or len(yolo_results[0].keypoints.xy) == 0
        ):
            print(f"No YOLO detection/keypoints for: {image_name}")
            continue

        r = yolo_results[0]

        keypoints_xy = r.keypoints.xy[0].cpu().numpy()
        keypoints_conf = r.keypoints.conf[0].cpu().numpy()

        rvec, tvec = processor.CalculatePoseFromKeypoints(
            keypoints_xy,
            keypoints_conf
        )

        if rvec is None or tvec is None:
            print(f"PnP failed for: {image_name}")
            continue

        pnp_center_2d = project_pnp_center_to_image(
            rvec,
            tvec,
            processor.camera_matrix,
            processor.dist_coeffs
        )

        pnp_x, pnp_y = pnp_center_2d
        gt_x, gt_y = gt_center_2d

        pixel_error = np.linalg.norm(
            np.array([pnp_x, pnp_y]) - np.array([gt_x, gt_y])
        )

        # PnP tvec is in cm if your UGV 3D points are in cm
        pnp_center_m = np.array(tvec).flatten() / 100.0
        pnp_z_m = pnp_center_m[2]

        # Convert pixel error to approximate meters
        fx = processor.camera_matrix[0, 0]

        meters_per_pixel = pnp_z_m / fx

        metric_error_m = pixel_error * meters_per_pixel

        gt_height_m = heights[image_name]

        z_error_m = abs(pnp_z_m - gt_height_m)

        error_3d_approx_m = np.sqrt(metric_error_m**2 + z_error_m**2)

        print("\n==============================")
        print(image_name)
        print("==============================")
        print(f"GT center 2D:  x={gt_x:.1f}, y={gt_y:.1f}")
        print(f"PnP center 2D: x={pnp_x:.1f}, y={pnp_y:.1f}")
        print(f"2D pixel error: {pixel_error:.2f} px")
        print(f"Approx 2D metric error: {metric_error_m:.3f} m")
        print(f"GT height: {gt_height_m:.3f} m")
        print(f"PnP z:     {pnp_z_m:.3f} m")
        print(f"Z error:   {z_error_m:.3f} m")
        print(f"Approx 3D error: {error_3d_approx_m:.3f} m")

        # Draw result
        img_draw = r.plot()

        cv2.circle(img_draw, (int(gt_x), int(gt_y)), 5, (255, 0, 255), -1)
        cv2.putText(
            img_draw,
            "GT center",
            (int(gt_x) + 10, int(gt_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

        cv2.circle(img_draw, (int(pnp_x), int(pnp_y)), 5, (0, 255, 255), -1)
        cv2.putText(
            img_draw,
            "PnP center",
            (int(pnp_x) + 10, int(pnp_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        screen_width = 1280
        screen_height = 720
        h, w = img_draw.shape[:2]
        scale = min(screen_width / w, screen_height / h)
        resized = cv2.resize(img_draw, (int(w * scale), int(h * scale)))

        cv2.imshow("PnP vs manual GT center", resized)
        cv2.waitKey(0)

        results_summary.append({
            "image": image_name,
            "pixel_error": pixel_error,
            "metric_error_m": metric_error_m,
            "z_error_m": z_error_m,
            "error_3d_approx_m": error_3d_approx_m,
        })

    cv2.destroyAllWindows()

    if len(results_summary) > 0:
        pixel_errors = [r["pixel_error"] for r in results_summary]
        z_errors = [r["z_error_m"] for r in results_summary]
        metric_errors = [r["metric_error_m"] for r in results_summary]
        approx_3d_errors = [r["error_3d_approx_m"] for r in results_summary]

        print("\n==============================")
        print("FINAL SUMMARY")
        print("==============================")
        print(f"Images evaluated: {len(results_summary)}")
        print(f"Mean 2D pixel error: {np.mean(pixel_errors):.2f} px")
        print(f"Mean 2D metric error: {np.mean(metric_errors):.3f} m")
        print(f"Mean Z error: {np.mean(z_errors):.3f} m")
        print(f"Mean approx 3D error: {np.mean(approx_3d_errors):.3f} m")
        print(f"Max 2D pixel error: {np.max(pixel_errors):.2f} px")
        print(f"Max 2D metric error: {np.max(metric_errors):.3f} m")
        print(f"Max Z error: {np.max(z_errors):.3f} m")
        print(f"Max approx 3D error: {np.max(approx_3d_errors):.3f} m")


if __name__ == "__main__":
    main()