import cv2
import numpy as np

# =========================
# 1. 8 st 3D-punkter
# =========================
object_points_dict = {
    1: [-0.2,  0.2, 0.0],  # vänster fram
    2: [ 0.2,  0.2, 0.0],  # höger fram
    3: [ 0.2, -0.2, 0.0],  # höger bak
    4: [-0.2, -0.2, 0.0],  # vänster bak

    5: [ 0.0,  0.2, 0.0],  # mitten fram
    6: [ 0.0, -0.2, 0.0],  # mitten bak
    7: [ 0.2,  0.0, 0.0],  # höger mitten
    8: [-0.2, 0.0, 0.0],   # vänster mitten
}

# =========================
# 2. Kamera
# =========================
fx, fy = 800, 800
cx, cy = 320, 240

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

dist_coeffs = np.zeros((4,1))

# =========================
# 3. State
# =========================
image_points = []
object_points = []

selected_id = None
frame = None

# =========================
# 4. Load image
# =========================
frame = cv2.imread("PNPtestbildSNE.jpg")
if frame is None:
    print("Kunde inte läsa bilden")
    exit()

clone = frame.copy()

# =========================
# 5. Mouse click
# =========================
def click_event(event, x, y, flags, param):
    global image_points, object_points, selected_id, frame

    if event == cv2.EVENT_LBUTTONDOWN and selected_id is not None:
        print(f"ID {selected_id} -> pixel {x,y}")

        image_points.append([x, y])
        object_points.append(object_points_dict[selected_id])

        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(selected_id), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        selected_id = None

# =========================
# 6. Axlar
# =========================
def draw_axes(img, rvec, tvec):
    axis = np.float32([
        [0,0,0],
        [0.1,0,0],
        [0,0.1,0],
        [0,0,0.1]
    ])

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec,
                                 camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)

    o, x, y, z = imgpts.reshape(-1,2)

    cv2.line(img, tuple(o), tuple(x), (0,0,255), 3)
    cv2.line(img, tuple(o), tuple(y), (0,255,0), 3)
    cv2.line(img, tuple(o), tuple(z), (255,0,0), 3)

# =========================
# 7. Confidence (reprojection error)
# =========================
def compute_confidence(rvec, tvec, obj_pts, img_pts):
    proj, _ = cv2.projectPoints(
        np.array(obj_pts, dtype=np.float32),
        rvec, tvec, camera_matrix, dist_coeffs
    )

    proj = proj.reshape(-1,2)
    img_pts = np.array(img_pts, dtype=np.float32)

    err = np.linalg.norm(proj - img_pts, axis=1)
    rms = np.sqrt(np.mean(err**2))

    confidence = max(0.0, 100 - rms)  # enkel mapping

    return rms, confidence

# =========================
# 8. UI
# =========================
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", click_event)

print("""
Controls:
1-8 = välj vilken 3D-punkt du ska klicka
p   = run PnP
c   = reset
ESC = exit
""")

while True:
    key = cv2.waitKey(0)

    # välj punkt
    if key in [ord(str(i)) for i in range(1,9)]:
        selected_id = int(chr(key))
        print("Vald 3D-id:", selected_id)

    # run PnP
    if key == ord('p'):
        if len(image_points) >= 4:
            img_pts = np.array(image_points, dtype=np.float32)
            obj_pts = np.array(object_points, dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                camera_matrix,
                dist_coeffs
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)

                forward = R[:,2]
                print("\nForward vector:", forward)

                rms, conf = compute_confidence(
                    rvec, tvec, object_points, image_points
                )

                print("Reprojection RMS error:", rms)
                print("Confidence:", conf, "%")

                draw_axes(frame, rvec, tvec)
                cv2.imshow("Frame", frame)

            else:
                print("PnP fail")

    # reset
    if key == ord('c'):
        image_points = []
        object_points = []
        frame = clone.copy()
        cv2.imshow("Frame", frame)

    if key == 27:
        break

cv2.destroyAllWindows()