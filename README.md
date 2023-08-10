# CSSE463_Project_Car_Lane_Detection
Lane Detection






Vehicle Detection

Ultralytics
YOLO
website:https://docs.ultralytics.com/
pip install ultralytics

install pytorch
https://pytorch.org/get-started/locally/


file structure for training dataset under same folder:
[path]/images/       store all images
[path]/labels/       store all labels in .txt file corresponding to images

YOLO label format:
(detection-object-index x-center-ratio y-center-ratio width-ratio height-ratio)
example label file content, where first line is the label of object "0" with box of whole image, second is box of object 1 in up left quarter
0 0.5 0.5 1 1
1 0.25 0.25 0.5 0.5

name of classes for yolo model
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


files in YOLO folders:
a.yaml: Used for training. Store the path to training set and validate set, the number of class as "nc", names of class

label.py: label the images in the folder. Please create the "labels" folder parallel to "images" folder before executing. Confidence level can be adjusted

train.py: train the model with configuration in .yaml file

predict.py: generate the image with boxes based on model using


dataset used: https://ieee-dataport.org/open-access/mit-driveseg-semi-auto-dataset