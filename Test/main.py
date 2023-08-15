import torch
import cv2
from data.MyDataSet import MyDataSet
from torch.utils.data import DataLoader
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def predictImage(imgPath, device, U_Net_Model, YOLO_Model):
    imgSize = (512, 512)
    inputImg = cv2.imread(imgPath)
    inputImg = cv2.resize(inputImg, imgSize)
    tmp_inputImg = inputImg
    tmp_inputImg = cv2.cvtColor(tmp_inputImg, cv2.COLOR_BGR2GRAY)
    tmp_inputImg = tmp_inputImg.reshape(1, tmp_inputImg.shape[0], tmp_inputImg.shape[1])
    tmp_inputImg = torch.from_numpy(tmp_inputImg)
    tmp_inputImg = tmp_inputImg.unsqueeze(0)
    tmp_inputImg = tmp_inputImg.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred = U_Net_Model(tmp_inputImg)
        
        # prepare for storage
        pred = np.array(pred.data.cpu()[0])[0]

        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        non_zero_indices = np.nonzero(pred)
        non_zero_values = pred[non_zero_indices]

        mask = np.zeros_like(inputImg)

        # Set the labeled points to a color (e.g., red)
        color = (0, 255, 0)  # BGR format
        for y, x in zip(non_zero_indices[0], non_zero_indices[1]):
            mask[y, x] = color


    line_mask=linedetect(mask)
    
    yoloResult = YOLO_Model.predict(inputImg,imgsz=512, conf=0.1)  # predict on an image
    yolo_mask = np.zeros_like(inputImg)
    color=(0,128,255)
    r=yoloResult[0]
    labelDict=r.names
    i=1
    for box in r:
        data=box.boxes.data[0]
        x1=int(data[0])
        y1=int(data[1])
        x2=int(data[2])
        y2=int(data[3])
        
        p=round(float(data[4]),3)
        label=labelDict[int(data[5])]
        text=str(i)+" "+label+" "+str(p)
        distance=imgSize[0]/(x2-x1)
        text+=" "+str(round(distance,3))+" m"
        cv2.rectangle(yolo_mask,(x1,y1),(x2,y2),color,3)
        cv2.putText(yolo_mask,text,(20,i*25),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)
        cv2.putText(yolo_mask,str(i),(x1,y2),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)
        i+=1
    
    #select the mask to plot
    #mask: line
    #yolo_mask: boxes and text about type, probability, and distance
    #line_mask: lines generated through hough transform and produced vanish point
    mask=yolo_mask+mask+line_mask
    
    # plt.imshow(mask)
    # plt.show()
    
    # cv2.imwrite("./inputImg.jpg", inputImg)
    # cv2.imwrite("./U_Net_Output.jpg", pred)
    # cv2.imwrite("./U_Net_Combined.jpg", cv2.addWeighted(inputImg, 1, mask, 0.5, 0))
    # cv2.imwrite("./YOLO_Output.jpg", im_array)

    # Add the mask to the original image
    return cv2.addWeighted(inputImg, 1, mask, 0.8, 0)  # Adjust the alpha value for transparency

def linedetect(mask):
    gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 100, minLineLength=50,maxLineGap=50)
    image=np.zeros_like(mask)
    vanish_x=0
    vanish_y=0
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(image,(x1,y1),(x2,y2),(255,0,255),1)
        if len(lines)>2:
            lines1=[]
            for line in lines:
                x1,y1,x2,y2=line[0]
                a=(y2-y1)/(x2-x1) if (x2-x1)!=0 else 1000000
                lines1.append((a,y1-x1*a))#a b where y=ax+b
            
            intersections=[]
            for line1 in lines1:
                a1,b1=line1
                for line2 in lines1:
                    a2,b2=line2
                    if not -0.1<a1-a2<0.1:#ignore similar gradient lines
                        x=(b2-b1)/(a1-a2)
                        y=a1*x+b1
                        intersections.append((x,y))
            if (n:=len(intersections))>0:
                xsum=0
                ysum=0
                for x,y in intersections:
                    xsum+=x
                    ysum+=y
                xaverage=xsum/n
                yaverage=ysum/n
                #remove faraway points
                for x,y in intersections:
                    if (x-xaverage)**2+(y-yaverage)**2>100:
                        xsum-=x
                        ysum-=y
                        n-=1
                vanish_x=xaverage=int(xsum/n)
                vanish_y=yaverage=int(ysum/n)
                cv2.circle(image,(xaverage,yaverage),20,(255,255,255),1)
                cv2.line(image,(vanish_x-32,vanish_y),(vanish_x+32,vanish_y),(255,255,255),1)
                cv2.line(image,(vanish_x,vanish_y-32),(vanish_x,vanish_y+32),(255,255,255),1)
    # plt.imshow(image)
    # plt.show()
    return image
    

YOLO_Path = "./weights/best.pt"
U_Net_Path = "./model.mod"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
U_Net_Model = torch.load(U_Net_Path,map_location='cpu')
U_Net_Model.eval()
U_Net_Model = U_Net_Model.to(device)
# Load a model
YOLO_Model = YOLO(YOLO_Path)  # load a pretrained model (recommended for training)
# YOLO_Model=YOLO('yolov8n.pt')


imgPath = "./images/test.jpg"
result = predictImage(imgPath, device, U_Net_Model, YOLO_Model)
# Save the result
output_path = "combined1.jpg"
cv2.imwrite(output_path, result)


'''
ROOT_PATH ="./images/images/"
OUTPUT_ROOT_PATH = "./images/output/"
imgListPath = "./images/images/imgList.txt"
with open(imgListPath) as f:
    imgPaths = [line.rstrip('\n') for line in f]

for imgPath in imgPaths:
    inputPath = os.path.join(ROOT_PATH, imgPath)
    outputPath = os.path.join(OUTPUT_ROOT_PATH, imgPath)
    result = predictImage(inputPath, device, U_Net_Model, YOLO_Model)
    cv2.imwrite(outputPath, result)


# Set the path to the directory containing your images
images_directory = './images/output/'

# Get the list of image filenames in the directory
image_filenames = sorted(os.listdir(images_directory))

# Specify the output video filename
output_video_filename = 'output_video.mp4'

# Define the frame size (width, height) for the video
frame_size = (512, 512)  # Adjust the size as needed

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_filename, fourcc, 30, frame_size)

# Loop through image filenames and write frames to the video
for image_filename in image_filenames:
    image_path = os.path.join(images_directory, image_filename)
    image = cv2.imread(image_path)
    if image is not None:
        out.write(image)

# Release the video writer
out.release()

print("Video created:", output_video_filename)
'''