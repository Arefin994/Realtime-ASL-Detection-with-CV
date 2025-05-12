import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0) 
detector = HandDetector(detectionCon=0.8, maxHands=1)

offset = 20
imSize = 300

folder = "Data/C"

cnt = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) 
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imWhite = np.ones((imSize, imSize, 3), np.uint8) * 255
        # cv2.rectangle(img, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 2)

        imCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        
        
        aspectRatio = h/w

        if aspectRatio > 1:
            k = imSize / h
            wCal = math.ceil(k * w)
            imResize = cv2.resize(imCrop, (wCal, imSize))
            wGap = math.ceil((imSize - wCal) / 2)
            imWhite[:, wGap:wCal+wGap] = imResize
        else:
            k = imSize / w
            hCal = math.ceil(k * h)
            imResize = cv2.resize(imCrop, (imSize, hCal))
            hGap = math.ceil((imSize - hCal) / 2)
            imWhite[hGap:hCal+hGap, :] = imResize


        cv2.imshow("Crop", imCrop)
        cv2.imshow("imWhite", imWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cnt += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imWhite)
        print(cnt)