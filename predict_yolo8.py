from ultralytics import YOLO

model = YOLO('ultralytics/runs/detect/train2/weights/best.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('test/000003.jpg', save=True, imgsz=640, conf=0.5)

# results = model('test/000003.jpg')
# for result in results:
#     boxes = result.boxes
#     print(boxes)