from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.pt")  # take pretrained model



# Use the model
model.train(data="data_config.yaml", epochs=10, imgsz=1024, pretrained=True)  # train the model