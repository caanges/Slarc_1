import cv2
import numpy as np

# =========================
# 1. 3D PUNKTER
# (lite bättre numerisk stabilitet)
# =========================
object_points_dict = {
    1: [-0.2,  0.2, 0.0],
    2: [ 0.2,  0.2, 0.0],
    3: [ 0.2, -0.2, 0.0],
    4: [-0.2, -0.2, 0.0],
    5: [ 0.0,  0.2, 0.0],
    6: [ 0.0, -0.2, 0.0],
    7: [ 0.2,  0.0, 0.0],
    8: [-0.2,  0.0, 0.0],
}

# =========================
# 2. CAMERA
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
# 3. STATE
# =========================
image_points = []
object_points = []
selected_id = None

frame = cv2.imread("PNPtestbildSNE.jpg")
if frame is None:
    raise Exception("Kunde inte läsa bilden")

clone = frame.copy()

# =========================
# 4. CLICK
# =========================
def click_event(event, x, y, flags, param):
    global selected_id, image_points, object_points, frame

    if event == cv2.EVENT_LBUTTONDOWN and selected_id is not None:
        print(f"ID {selected_id} -> {x},{y}")

        image_points.append([x, y])
        object_points.append(object_points_dict[selected_id])

        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(selected_id), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        selected_id = None

# =========================
# 5. AXES
# =========================
def draw_axes(img, rvec, tvec):
    axis = np.float32([
        [0,0,0],
        [0.1,0,0],
        [0,0.1,0],
        [0,0,0.1]
    ])

    imgpts, _ = cv2.projectPoints(
        axis, rvec, tvec, camera_matrix, dist_coeffs
    )

    imgpts = imgpts.astype(int).reshape(-1,2)
    o, x, y, z = imgpts

    cv2.line(img, tuple(o), tuple(x), (0,0,255), 3)
    cv2.line(img, tuple(o), tuple(y), (0,255,0), 3)
    cv2.line(img, tuple(o), tuple(z), (255,0,0), 3)

# =========================
# 6. PNP (ROBUST VERSION)
# =========================
def run_pnp():
    global frame

    if len(image_points) < 4:
        print("Behöver minst 4 punkter")
        return

    img_pts = np.array(image_points, dtype=np.float32)
    obj_pts = np.array(object_points, dtype=np.float32)

    # =========================
    # RANSAC STEP
    # =========================
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        iterationsCount=500,
        reprojectionError=12.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # =========================
    # FALLBACK (VIKTIG)
    # =========================
    if not success or inliers is None:
        print("RANSAC failed → fallback to solvePnP IPPE")

        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        inliers = np.arange(len(img_pts)).reshape(-1,1)

    # =========================
    # RESULT
    # =========================
    print("\n===== RESULTAT =====")

    R, _ = cv2.Rodrigues(rvec)
    forward = R[:,2]

    print("Forward:", forward)
    print("Inliers:", len(inliers), "/", len(img_pts))

    # =========================
    # VISUALISERING
    # =========================
    vis = frame.copy()
    draw_axes(vis, rvec, tvec)

    inlier_set = set(inliers.flatten())

    for i, (x, y) in enumerate(img_pts):
        if i in inlier_set:
            cv2.circle(vis, (int(x), int(y)), 6, (0,255,0), 2)
        else:
            cv2.circle(vis, (int(x), int(y)), 6, (0,0,255), 2)

    cv2.imshow("Result", vis)

# =========================
# 7. RESET
# =========================
def reset():
    global image_points, object_points, frame, selected_id
    image_points = []
    object_points = []
    selected_id = None
    frame = clone.copy()
    cv2.imshow("Frame", frame)

# =========================
# 8. UI
# =========================
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", click_event)

print("""
CONTROLS:
1-8 = välj punkt
p   = run PnP
c   = reset
ESC = exit
""")

while True:
    key = cv2.waitKey(0)

    if key in [ord(str(i)) for i in range(1,9)]:
        selected_id = int(chr(key))
        print("Vald:", selected_id)

    elif key == ord('p'):
        run_pnp()

    elif key == ord('c'):
        reset()

    elif key == 27:
        break

cv2.destroyAllWindows()