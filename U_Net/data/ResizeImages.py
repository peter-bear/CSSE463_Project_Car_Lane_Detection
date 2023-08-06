import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import threading


def resize_img(img):
    return cv2.resize(img, (512, 512))

def save_image(fromIdx, toIdx, ROOT, datalist):    
    for i in range(fromIdx, toIdx):
        image_path, label_path = get_label(datalist, idx=i)
        image_path =  os.path.join(ROOT, image_path)
        label_path = os.path.join(ROOT, label_path)
        img = plt.imread(image_path)
        annotation = load_json(label_path) 
        lebel_img_path = label_path.replace('.json', '.jpg')
        lable_image = generate_seg_label(annotation)
        img = resize_img(img)
        lable_image = resize_img(lable_image)

        print("Save File to "+lebel_img_path)
        cv2.imwrite(lebel_img_path, lable_image)

        print("Save File to "+image_path)
        cv2.imwrite(image_path, img)

def create_multithread_save_files(ROOT, FILELIST, THREAD_NUM):
    DATALIST_PATH = os.path.join(ROOT, FILELIST)
    # Load datalist
    with open(DATALIST_PATH) as f:
        datalist = [line.rstrip('\n') for line in f]  

    threads = []
    datalistLength = len(datalist)
    for i in range(THREAD_NUM):
        threads.append(threading.Thread(target=save_image, args=(i*datalistLength//6, (i+1)*datalistLength//6, ROOT, datalist)))
        threads[i].start()

    for i in range(THREAD_NUM):
        threads[i].join()

    print("Finish")


def get_label(datalist, idx):
    """
    returns the corresponding label path for each image path
    """ 
    image_path = datalist[idx]
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.json')
    return image_path, label_path

def load_json(label_path):
    with open(label_path, "r") as f:
        annotation = json.load(f)
    return annotation


if __name__ == '__main__':
    testAddr = "./SDLane/test/"
    testList = 'test_list.txt'
    trainAddr = "./SDLane/train/"
    trainList = 'train_list.txt'

    create_multithread_save_files(testAddr, testList, 24)
    create_multithread_save_files(trainAddr, trainList, 24)