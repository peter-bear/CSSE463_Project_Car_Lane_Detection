# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

path=''#path above "images" folder and "labels" folder

filenames=[file for file in os.listdir(path+'/images')]
a=model.predict([path+'/images/'+file for file in filenames],save=False, imgsz=320, conf=0.1)#the confidence threshold of object

#convert ultralytics label to customized
targets={0:0,1:1,2:2,3:3,5:4,7:5}#person bicycle car motorcycle bus truck in yolov8n.pt model
for j in range(len(a)):
    b=a[j]
    for i in range(len(b.boxes.cls)):
        if int(b.boxes.cls[i]) in targets:
            print(path+"/labels/"+filenames[j],int(b.boxes.cls[i]))
            with open(path+"/labels/"+filenames[j].replace(".png","")+".txt",'a') as txt:
                txt.write(str(targets[int(b.boxes.cls[i])])+" "+str(b.boxes.xywhn.tolist()[i])[2:-2].replace(",","")+"\n")
                