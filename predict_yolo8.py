from ultralytics import YOLO

model = YOLO('ultralytics/runs/detect/train2/weights/best_small.pt')

# Run inference on test.json image
model.predict('test.json/000003.jpg', save=True, imgsz=640, conf=0.5)

