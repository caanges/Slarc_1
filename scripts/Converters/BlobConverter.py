import blobconverter

blob_path = blobconverter.from_onnx(
    model="yolov8n-pose.onnx",
    data_type="FP16",
    shaves=6,
    use_cache=True
)
print(blob_path)