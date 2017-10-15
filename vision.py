#!/usr/bin/env python
import rospy
import math
import cv2
from PIL import Image
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils

def isPerson(img):
	par=0
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = imutils.resize(img, width=min(400, img.shape[1]))
    orig = img.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
       padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
       cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        par+=1
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    if(par==1):
        x=False
        return True
    else:
        return False

def isFace(img):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.2, 5)
	picList=[]
	for (x,y,w,h) in faces:
		xC=x
		yC=y
		width=w
		height=h
		a = np.zeros([width,height,3],dtype=np.uint8)
		for i in range (0,width):
			for j in range (0,height):
				pic[x][y]=gray[i+xC][j+yC]
		picList.append(pic)
	if (xC==0):
		return picList,False
	else:
		print "Face found..."
		return picList,True