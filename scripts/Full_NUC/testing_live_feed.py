import cv2
import depthai as dai
import numpy as np


# =========================
# CHANGE THESE
# =========================
BLOB_PATH = r"C:\Users\egn23014\Downloads\best.rvc2_legacy.rvc2\best.blob"


INPUT_SIZE = 640
NUM_KEYPOINTS = 13

CONF_THRES = 0.25
KPT_CONF_THRES = 0.25
NMS_IOU_THRES = 0.5



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def box_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0

    return inter_area / union


def nms_detections(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)

        detections = [
            det for det in detections
            if box_iou(best["box"], det["box"]) < iou_threshold
        ]

    return kept


def decode_multi_output_yolov8_pose(outputs):
    all_detections = []

    pairs = [
        ("output1_yolov8", "kpt_output1", 80, 80, 8),
        ("output2_yolov8", "kpt_output2", 40, 40, 16),
        ("output3_yolov8", "kpt_output3", 20, 20, 32),
    ]

    for box_name, kpt_name, grid_w, grid_h, stride in pairs:
        box_raw = outputs[box_name]
        kpt_raw = outputs[kpt_name]

        num_cells = grid_w * grid_h

        box_values = box_raw.size // num_cells
        kpt_values = kpt_raw.size // num_cells

        print()
        print(box_name, "box values per cell:", box_values)
        print(kpt_name, "kpt values per cell:", kpt_values)

        boxes = box_raw.reshape(box_values, num_cells).T
        keypoints_raw = kpt_raw.reshape(kpt_values, num_cells).T

        for i in range(num_cells):
            pred = boxes[i]

            obj_conf = float(pred[4])

            if box_values > 5:
                class_conf = float(pred[5])
                conf = obj_conf * class_conf
            else:
                conf = obj_conf

            if conf < CONF_THRES:
                continue

            grid_x = i % grid_w
            grid_y = i // grid_w

            cx = (grid_x + 0.5) * stride
            cy = (grid_y + 0.5) * stride

            left = float(pred[0]) * stride
            top = float(pred[1]) * stride
            right = float(pred[2]) * stride
            bottom = float(pred[3]) * stride

            x1 = int(cx - left)
            y1 = int(cy - top)
            x2 = int(cx + right)
            y2 = int(cy + bottom)

            x1 = max(0, min(INPUT_SIZE, x1))
            y1 = max(0, min(INPUT_SIZE, y1))
            x2 = max(0, min(INPUT_SIZE, x2))
            y2 = max(0, min(INPUT_SIZE, y2))

            kpts = []
            kpt_data = keypoints_raw[i]

            for k in range(NUM_KEYPOINTS):
                kx = float(kpt_data[k * 3])
                ky = float(kpt_data[k * 3 + 1])
                kc = float(kpt_data[k * 3 + 2])
                kpts.append((kx, ky, kc))

            all_detections.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "keypoints": kpts
            })

    all_detections = sorted(all_detections, key=lambda d: d["conf"], reverse=True)

    print()
    print("==============================")
    print("RAW DETECTIONS BEFORE NMS:", len(all_detections))
    print("==============================")

    for i, det in enumerate(all_detections[:10]):
        print(f"Detection {i}: conf={det['conf']:.4f}, box={det['box']}")

    all_detections = nms_detections(all_detections, iou_threshold=NMS_IOU_THRES)

    print()
    print("==============================")
    print("DETECTIONS AFTER NMS:", len(all_detections))
    print("==============================")

    for i, det in enumerate(all_detections):
        print(f"\nDetection {i}")
        print(f"Confidence: {det['conf']:.4f}")
        print(f"Box: {det['box']}")

        for k, (kx, ky, kc) in enumerate(det["keypoints"]):
            print(f"  Keypoint {k}: x={kx:.1f}, y={ky:.1f}, conf={kc:.4f}")

    return all_detections


def draw_detections(image, detections):
    result = image.copy()

    if len(detections) == 0:
        cv2.putText(
            result,
            "No detections",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return result

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        keypoints = det["keypoints"]

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            result,
            f"UGV {conf:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        for i, (kx, ky, kc) in enumerate(keypoints):
            kx = int(kx)
            ky = int(ky)

            if kc >= KPT_CONF_THRES:
                cv2.circle(result, (kx, ky), 5, (0, 0, 255), -1)
                cv2.putText(
                    result,
                    str(i),
                    (kx + 6, ky - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                )

    return result


def main():
    pipeline = dai.Pipeline()

    # =========================
    # MONO CAMERA
    # =========================
    mono = pipeline.create(dai.node.MonoCamera)
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    # =========================
    # IMAGE MANIP
    # Convert mono frame to 640x640 NN input
    # NOTE: this resizes, not true letterbox
    # =========================
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResizeThumbnail(INPUT_SIZE, INPUT_SIZE)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)

    manip.setMaxOutputFrameSize(1228800)

    # =========================
    # NEURAL NETWORK
    # =========================
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(BLOB_PATH)

    # =========================
    # OUTPUTS TO PC/NUC
    # =========================
    xout_frame = pipeline.create(dai.node.XLinkOut)
    xout_frame.setStreamName("frame")

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    # =========================
    # LINK PIPELINE
    # =========================
    mono.out.link(manip.inputImage)
    manip.out.link(nn.input)

    # Send manipulated frame to PC for drawing
    manip.out.link(xout_frame.input)

    # Send NN result to PC
    nn.out.link(xout_nn.input)

    with dai.Device(pipeline) as device:
        frame_q = device.getOutputQueue("frame", maxSize=4, blocking=False)
        nn_q = device.getOutputQueue("nn", maxSize=4, blocking=False)

        while True:
            frame_msg = frame_q.get()
            nn_data = nn_q.get()

            shown_image = frame_msg.getCvFrame()

            outputs = {}

            for name in nn_data.getAllLayerNames():
                data = np.array(nn_data.getLayerFp16(name), dtype=np.float32)
                outputs[name] = data

            detections = decode_multi_output_yolov8_pose(outputs)

            result = draw_detections(shown_image, detections)

            cv2.imshow("Live YOLOv8 pose on OAK-D SR", result)

            if cv2.waitKey(1) == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()