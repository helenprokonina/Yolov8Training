from ultralytics import YOLO

model = YOLO('ultralytics/runs/detect/train2/weights/best.pt')

# Run inference on test image
model.predict('test/000003.jpg', save=True, imgsz=640, conf=0.5)

