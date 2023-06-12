from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

# get video from file
cap = cv.VideoCapture("traffOne.avi")   # video

# use this for a live webcam
#cap = cv.VideoCapture(0)
#cap.set(3, 1920)
#cap.set(4, 1080)

# get frames per second of video, may not work for webcam
fps = cap.get(cv.CAP_PROP_FPS)
print('fps ', fps)

# pre-trained weights for object detection
model = YOLO("yolov8l.pt")

# mask image that cuts out non relevant portions of images
mask = cv.imread("traffMaskTwo.png")

# list of detectable object names
f = open("coco.names", "r")
namesStr = f.read()
listNames = namesStr.split()
print("000   listNames ", listNames)

# index the main loop, used for testing
loopIndex = -1

# Tracker
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

# the lines that the cars cross
lineCrossCoordsRight = [ 760, 540, 1540, 540 ]  # x1, y1, x2, y2
lineCrossCoordsLeft = [ 100, 440, 720, 440 ]  # x1, y1, x2, y2

# used for counting vehicles
totalCountRight = []
totalCountLeft = []

while True:
    loopIndex += 1
    # read in frame
    success, img = cap.read()
    # set region to detect vehicles
    imgRegion = cv.bitwise_and(img, mask)
    # get bounding boxes for detected objects
    results = model(imgRegion, stream = True)
    # detections will be made into an np array, this is for tracking
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:   # iter through bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1    # calc width and height
            print(x1, y1, x2, y2)
            conf = math.ceil((box.conf[0] * 100)) / 100   # confidence rounded
            print("conf ", conf)
            clss = int(box.cls[0])    # index of class of object in box
            currentClass = listNames[clss]   # string name of object in box
            print("currentClass ", currentClass)

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                currentArray = np.array([ x1, y1, x2, y2, conf ])
                # create a np array for tracking
                detections = np.vstack((detections, currentArray))

    # for tracking, list of boxes
    resultsTracker = tracker.update(detections)
    # draw right cross counting line
    cv.line(img, (lineCrossCoordsRight[0], lineCrossCoordsRight[1]), (lineCrossCoordsRight[2], lineCrossCoordsRight[3]), (0, 0, 255), 5)
    # draw left cross counting line
    cv.line(img, (lineCrossCoordsLeft[0], lineCrossCoordsLeft[1]), (lineCrossCoordsLeft[2], lineCrossCoordsLeft[3]),
            (0, 0, 255), 5)

    for result in resultsTracker:
        # each object now has position and ID for tracking
        x1, y1, x2, y2, idd = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        idd = int(idd)
        w, h = x2 - x1, y2 - y1   # calc width and height of box
        print("tracker result ", result)

        # draw boxes
        cvzone.cornerRect(img, (x1, y1, w, h), l = 10, rt = 2, colorR = (255, 0, 0))
        cvzone.putTextRect(img, f'{currentClass} {idd}', (max(0, x1), max(20, y1)), 
                               scale = 1.2, thickness = 1, offset = 3)
        # calc center of boxes, centers are the points that cross the counting line
        cx, cy = x1 + w // 2, y1 + h // 2
        # draw centers of detected vehicles
        cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)
        print("cx, cy ", cx, cy)

        # if center of vehicle is in the counting region
        # right
        if lineCrossCoordsRight[0] < cx < lineCrossCoordsRight[2] and lineCrossCoordsRight[1] - 20 < cy < lineCrossCoordsRight[1] + 20:
            # if vehicle has not been counted already
            if totalCountRight.count(idd) == 0:
                totalCountRight.append(idd)
                # turn line green when a vechicle is counted
                cv.line(img, (lineCrossCoordsRight[0], lineCrossCoordsRight[1]), (lineCrossCoordsRight[2], lineCrossCoordsRight[3]), (0, 255, 0), 5)
        # left
        elif lineCrossCoordsLeft[0] < cx < lineCrossCoordsLeft[2] and lineCrossCoordsLeft[1] - 20 < cy < lineCrossCoordsLeft[1] + 20:
            # if vehicle has not been counted already
            if totalCountLeft.count(idd) == 0:
                totalCountLeft.append(idd)
                # turn line green when a vechicle is counted
                cv.line(img, (lineCrossCoordsLeft[0], lineCrossCoordsLeft[1]), (lineCrossCoordsLeft[2], lineCrossCoordsLeft[3]), (0, 255, 0), 5)

        # draw counter text
        cvzone.putTextRect(img, "Left Side: " + str(len(totalCountLeft)) +
             "   Right Side: " + str(len(totalCountRight)), (50, 50))
                          
        
   
    # save video frames
    #cv.imwrite("TrafficFrames\\traff_" + str(loopIndex) + ".png" , img)
    # show video frames
    cv.imshow("Vehicle Counter", img)
    # show video frames with region
    #cv.imshow("Region", imgRegion)

    cv.waitKey(1)