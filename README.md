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


files in YOLO folders:
a.yaml: Used for training. Store the path to training set and validate set, the number of class as "nc", names of class

label.py: label the images in the folder. Please create the "labels" folder parallel to "images" folder before executing. Confidence level can be adjusted

train.py: train the model with configuration in .yaml file

predict.py: generate the image with boxes based on model using