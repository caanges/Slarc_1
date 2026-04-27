import cv2
import depthai as dai
from pathlib import Path

SAVE_DIR = Path(r"C:\Users\egn23014\bilder SLaRC") #change to where you want the pictures
SAVE_DIR.mkdir(exist_ok=True)

device = dai.Device()

with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    q = cam.requestOutput(size=(640, 480)).createOutputQueue()

    pipeline.start()

    image_number = 1

    print("Press ENTER to save image.")
    print("Press Q to quit.")

    while True:
        frame_msg = q.get()
        frame = frame_msg.getCvFrame()

        cv2.imshow("OAK-D SR Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            filename = SAVE_DIR / f"image{image_number:03d}.png"
            cv2.imwrite(str(filename), frame)
            print(f"Saved {filename}")
            image_number += 1

        elif key == ord("q"):
            break

cv2.destroyAllWindows()