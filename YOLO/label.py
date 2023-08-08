# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

path='./photos/photo50'


#predict
# for file in os.listdir(path):
#     model.predict(path+'\\'+file,save=True, imgsz=320, conf=0.1)
    


filenames=[file for file in os.listdir(path+'/images')]
a=model.predict([path+'/images/'+file for file in filenames],save=False, imgsz=320, conf=0.1)
# print([int(i) for i in a[0].boxes.cls])
# print(a[0].boxes.xywhn.tolist())

#convert ultralytics label to customized
targets={0:0,1:1,2:2,3:3,5:4,7:5}#person bicycle car motorcycle bus truck
for j in range(len(a)):
    b=a[j]
    for i in range(len(b.boxes.cls)):
        if int(b.boxes.cls[i]) in targets:
            print(path+"/labels/"+filenames[j],int(b.boxes.cls[i]))
            with open(path+"/labels/"+filenames[j].replace(".png","")+".txt",'a') as txt:
                txt.write(str(targets[int(b.boxes.cls[i])])+" "+str(b.boxes.xywhn.tolist()[i])[2:-2].replace(",","")+"\n")
                