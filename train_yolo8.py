import os
import glob

from ultralytics import YOLO

#Load a model
model = YOLO("yolov8m.pt")  # take pretrained model



# Use the model
model.train(data="data_config.yaml", epochs=20, imgsz=640, pretrained=True)  # train the model

