#!/usr/bin/env python3
import rospy
from utils.msg import ImgInfo
import numpy as np
import cv2 as cv
import time

def imgCallBack(info):
    t1 = time.time()
    # colorImg = (np.array(info.colorImg).reshape((480, 640, 3))).astype(np.uint8)
    # depImg = (np.array(info.depImg).reshape((480, 640)))
    colorImg = (np.array(info.colorImg).reshape((480, 640, 3))).astype(np.uint8)
    depImg = (np.array(info.depImg).reshape((480, 640)))
    print("time1: ",time.time()-t1)
    #rospy.loginfo("colorImg: %s, depthImg: %s",str(colorImg.shape),str(depImg.shape))
    #detect object - face for example
    gray = cv.cvtColor(colorImg, cv.COLOR_BGR2GRAY)
    t2 = time.time()
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)
    faceDis = []
    for (x, y, w, h) in faces:
        cv.rectangle(colorImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceDis.append(depImg[int((2*y+h)/2)][int((2*x+w)/2)])
    print("time2: ",time.time()-t2)
    cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
    cv.imshow('RealSense', colorImg)
    cv.namedWindow('DEPTH', cv.WINDOW_AUTOSIZE)
    cv.imshow('RealSenseDepth', depImg.astype(np.uint8))
    rospy.loginfo("face dis: %s",str(faceDis))
    key = cv.waitKey(1)
if __name__ == "__main__":
    rospy.init_node("trackObjet")
    sub = rospy.Subscriber("RealSenseInfo",ImgInfo,imgCallBack,queue_size=10)
    rospy.spin()