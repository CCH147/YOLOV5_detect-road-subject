from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

hog = cv2.HOGDescriptor() #導入opcv內建檢測模型
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread('test.jpg')                    # 讀取街道影像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 轉換成黑白影像


car_scan = cv2.CascadeClassifier("cars.xml") #導入模型
gray = cv2.medianBlur(gray, 1)   #模糊化灰階
cars = car_scan.detectMultiScale(gray, 1.1, 4 )  #調整參數
print(cars)

for (x, y, w, h) in cars: #繪製方框
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    text = "car"
    cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

found, ppl = hog.detectMultiScale(gray) #檢測行人
print(found , ppl)

for (x, y, w, h) in found: #繪製方框
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    text = "people"
    cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

bike_scan = cv2.CascadeClassifier("two_wheeler.xml") #導入二輪模型
bikes = bike_scan.detectMultiScale(gray, 1.4, 1 )  
print(bikes) #結果失敗

for (x, y, w, h) in bikes: #繪製方框
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
   
cv2.imshow('q1', img)
cv2.waitKey(0) # 按下任意鍵停止
cv2.destroyAllWindows()