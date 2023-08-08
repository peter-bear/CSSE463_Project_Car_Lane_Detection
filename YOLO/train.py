# -*- coding: utf-8 -*-
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='./a.yaml', epochs=100, imgsz=320, device='cpu')

