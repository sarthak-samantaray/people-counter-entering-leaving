import numpy as np
import math
import cvzone
import cv2
from ultralytics import YOLO
from sort import *

# Reading the video
cap = cv2.VideoCapture("videos/people.mp4")

# Model
model = YOLO("yolo_weights/yolov8l.pt")

# Tracker
tracker = Sort(max_age = 20, min_hits=3, iou_threshold=0.3)

# Limits 
limitsup = [103,161,296,161]
limitsdown = [527,489,735,489]

# Total_counts
total_count_up = []
total_count_down = []

# Mask
mask = cv2.imread("mask.png")

# Class names
class_names = ['person','bicycle','car','motorcycle','airplane','bus',
                'train',    'truck',    'boat',    'traffic light','fire hydrant',
                'stop sign',    'parking meter',    'bench',    'bird',    'cat',    
                'dog',    'horse',    'sheep',    'cow',    'elephant',    'bear',    
                'zebra',    'giraffe',    'backpack',    'umbrella',    'handbag',    
                'tie',    'suitcase',    'frisbee',    'skis',    'snowboard',    
                'sports ball',    'kite',    'baseball bat',    'baseball glove',    
                'skateboard',    'surfboard',    'tennis racket',    'bottle',    
                'wine glass',    'cup',    'fork',    'knife',    'spoon',    
                'bowl',    'banana',    'apple',    'sandwich',    'orange',    
                'broccoli',    'carrot',    'hot dog',    'pizza',    'donut',    
                'cake',    'chair',    'couch',    'potted plant',    'bed',    
                'dining table',    'toilet',    'tv',    'laptop',    'mouse',    
                'remote',    'keyboard',    'cell phone',    'microwave',    'oven',    
                'toaster',    'sink',    'refrigerator',    'book',    'clock',    
                'vase',    'scissors',    'teddy bear',    'hair drier',    
                'toothbrush']


while True:
    # Reading each frame
    success,img = cap.read()

    # Mask
    imgRegion = cv2.bitwise_and(img,mask)

    # Graphics

    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(730,260))

    # Results of each frame.
    results = model(imgRegion,stream=True)

    # Detections for tracking
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # BBOX
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            classes = int(box.cls[0])
            currentClass = class_names[classes]

            if currentClass=="person" and conf>0.5:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    results_tracker = tracker.update(detections)
    
    # lines
    cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,0,255),5)
    cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,0,255),5)

    for result in results_tracker:
        x1,y1,x2,y2,ID = result
        x1,y1,x2,y2,ID= int(x1),int(y1),int(x2),int(y2),int(ID)
        w,h = x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=3,rt=3,t=3,colorR=(255,0,255))
        cvzone.putTextRect(img,f"{currentClass}{ID}",(max(0,x1),max(35,y1)),scale=1,thickness=2)

        # making circles in the center of the bbox
        cx,cy = x1+w//2 , y1+h//2
        cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)

        # for going up
        if limitsup[0]<cx<limitsup[2] and limitsup[1]-15<cy<limitsup[3]+15:
            if total_count_up.count(ID)==0:
                total_count_up.append(ID)
                cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,255,0),5)

        if limitsdown[0]<cx<limitsdown[2] and limitsdown[1]-15<cy<limitsdown[3]+15:
            if total_count_down.count(ID)==0:
                total_count_down.append(ID)
                cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,255,0),5)
    cv2.putText(img,str(len(total_count_up)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img,str(len(total_count_down)),(1192,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)       


    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()