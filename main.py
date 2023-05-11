import cv2
import pyshine as ps
from ultralytics import YOLO
import cv2
import datetime
import random

cap=cv2.VideoCapture('path to video')
model=YOLO('path to model')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)   
size = (frame_width, frame_height)
video = cv2.VideoWriter('output1.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
while True:
    ret,frame=cap.read()
    results = model.predict(source=frame, conf=0.7)
    plot_one_box((0,860,1200,250),frame, color=(128,0,128),label='Height operation',line_thickness=3)
    plot_one_box((1700,670,2100,1900),frame, color=(128,128,0),label='Deflashing station',line_thickness=3)
    for f in results:
            boxes = f.boxes.xyxy
            category=f.boxes.cls
            confidence=f.boxes.conf
            for i,j,k in zip(boxes,category,confidence):
                xmin=i[0]
                ymin=i[1]
                xmax=i[2]
                ymax=i[3]
                
                if j==0:
                    plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(255,0,0),label='withHelmet',line_thickness=3)
                if  j==1:          
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(0,255,0),label='withoutHelmet',line_thickness=3)
                if j==2:
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(0,0,255),label='withGoggle',line_thickness=3)
                if j==3:
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(255,255,0),label='withoutGoggle',line_thickness=3)
                if j==4:
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(255,0,255),label='withGlove',line_thickness=3)
                if j==5:
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(0,255,255),label='withoutGlove',line_thickness=3)
                if j==6:
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(255,128,0),label='withoutShoe',line_thickness=3)
                if j==7:
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(255,0,128),label='withShoe',line_thickness=3)
                if j==8:
                     plot_one_box((int(xmin),int(ymin),int(xmax),int(ymax)),frame, color=(128,0,255),label='Mobile',line_thickness=3)

                
           
    video.write(frame)
    
