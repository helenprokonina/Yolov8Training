import os
import glob

from ultralytics import YOLO



import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolov8m.pt",
    config_path="data_config.yaml",
    confidence_threshold=0.4,
    device="cpu",
    image_size=640
)

for image_file in glob.glob("MOT20/test/MOT20-04/img1/*.jpg"):
    image = read_image_as_pil(image_file)
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=128,
        slice_width=128,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    result.export_visuals(export_dir="vis_result/")

