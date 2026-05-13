import cv2
import depthai as dai
from pathlib import Path

# =========================================================
# Create folder for videos
# =========================================================

BASE_DIR = Path(r"C:\Users\een23013\Slarc_1\scripts\Camera_to_pictures")

SAVE_DIR = BASE_DIR / "Videos_SLaRC"

if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)
    print(f"Created folder: {SAVE_DIR}")
else:
    print(f"Folder already exists: {SAVE_DIR}")

# =========================================================

device = dai.Device()

with dai.Pipeline(device) as pipeline:

    cam = pipeline.create(dai.node.Camera).build()

    q = cam.requestOutput(size=(640, 480)).createOutputQueue()

    pipeline.start()

    recording = False
    video_writer = None
    video_number = 1

    print("Press SPACE to start/stop recording.")
    print("Press Q to quit.")

    while True:

        frame_msg = q.get()
        frame = frame_msg.getCvFrame()

        # Recording indicator
        if recording:
            cv2.putText(
                frame,
                "REC",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        cv2.imshow("OAK-D SR Live Feed", frame)

        # Save frames while recording
        if recording and video_writer is not None:
            video_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF

        # =========================================================
        # SPACE = Start / Stop recording
        # =========================================================

        if key == 32:

            # START RECORDING
            if not recording:

                filename = SAVE_DIR / f"video{video_number:03d}.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                video_writer = cv2.VideoWriter(
                    str(filename),
                    fourcc,
                    30.0,
                    (640, 480)
                )

                recording = True

                print(f"Started recording: {filename}")

            # STOP RECORDING
            else:

                recording = False

                video_writer.release()
                video_writer = None

                print("Stopped recording.")

                video_number += 1

        # =========================================================
        # Quit
        # =========================================================

        elif key == ord("q"):
            break

    # Cleanup
    if video_writer is not None:
        video_writer.release()

cv2.destroyAllWindows()