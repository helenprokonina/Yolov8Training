import os
import random
import argparse
import cv2

from IPython.display import Image

import sahi

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil


from ultralytics import YOLO

from tracker import Tracker


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT for tracking")
    parser.add_argument(
        "--video_path", help="Path to MOTChallenge video",
        default=None, required=True)

    return parser.parse_args()

args = parse_args()


video_path = os.path.join('videos', args.video_path)

video_out_path = os.path.join('outs', args.video_path)

output_file = os.path.join('results', args.video_path.split(".")[0]+".txt")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolov8m.pt",
    config_path="data_config.yaml",
    confidence_threshold=0.4,
    device="cpu",
    image_size=640
)

#model = YOLO('ultralytics/runs/detect/train2/weights/best.pt')

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.5
frame_idx=0
out_results=[]


while ret:

    frame_idx+=1

    results = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=128,
        slice_width=128,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )


    results.export_visuals(export_dir="vis_result/")
    #
    # Image(f"vis_result/prediction_visual_{frame_idx}.png")

    #results = model(frame)

    results = results.to_coco_annotations()

    # print(results)

    for result in results:
        detections = []
        bbox = result['bbox']
        score = result['score']
        class_id = result['category_id']
        #for r in result.boxes.data.tolist():
            # x1, y1, x2, y2, score, class_id = r
        x1, y1, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x1 + width)
        y2 = int(y1 + height)
        class_id = int(class_id)
        if score > detection_threshold:
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            out_results.append([
                frame_idx, track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

#write results to .txt file
f = open(output_file, 'w')
for row in out_results:
    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
          (row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
