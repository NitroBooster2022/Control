#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Header
from utils.msg import Lane

class LaneDetector():
    def __init__(self):
        """
        Initialize the masks for histogram filter
        """
        print("Lane detection using histogram filter")
        self.maskh = np.zeros((480,640),dtype='uint8')
        h=int(0.8*480)
        polyh = np.array([[(0,h),(640,h),(640,480),(0,480)]]) # polyh might need adjustment
        cv2.fillPoly(self.maskh,polyh,255)
        self.masks = np.zeros((480,640),dtype='uint8')
        polys = np.array([[(0,300),(640,300),(640,340),(0,340)]]) # polys might need adjustment
        cv2.fillPoly(self.masks,polys,255)
        self.image = np.zeros((480,640))
        self.stopline = False
        """
        Initialize the lane follower node
        """
        rospy.init_node('lane_detector_node', anonymous=True)
        self.pub = rospy.Publisher("lane", Lane, queue_size=3)
        self.p = Lane()
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("automobile/image_raw", Image, self.image_callback)
        self.rate = rospy.Rate(15)

    def image_callback(self, data):
        """
        Callback function for the image processed topic
        :param data: Image data in the ROS Image format
        """
        # Update the header information
        header = Header()
        header.seq = data.header.seq
        header.stamp = data.header.stamp
        header.frame_id = data.header.frame_id
        self.p.header = header

        # Convert the image to the OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(data, "rgb8")

        # Extract the lanes from the image
        lanes = self.histogram(self.image, show=self.show)

        self.p.center = lanes

        #determine whether we arrive at intersection
        self.p.stopline = self.stopline

        # Publish the lane message
        self.pub.publish(self.p)

    def histogram(self, image, show=False):
        """
        Extract the lanes from the image using the histogram method
        :param image: Image to extract the lanes from
        :param show: Boolean to show the image
        :return: The steering angle
        """
        self.stopline = False
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h = 480
        w = 640
        img_roi = cv2.bitwise_and(img_gray,self.maskh)
        t = np.max(img_roi)-55
        if t<30:
            t=30
        np.clip(t,30,200)
        ret, thresh = cv2.threshold(img_roi, t, 255, cv2.THRESH_BINARY)
        hist=np.zeros((1,w))
        for i in range(w):
            hist[0,i]=np.sum(thresh[:,i])

        #stopline
        img_rois = cv2.bitwise_and(img_gray,self.masks)
        t = np.max(img_roi)-65
        if t<30:
            t=30
        np.clip(t,30,200)
        ret, threshs = cv2.threshold(img_rois, t, 255, cv2.THRESH_BINARY)
        hists=np.zeros((1,w))
        for i in range(w):
            hists[0,i]=np.sum(threshs[:,i])
        lanes=[]
        p=0
        for i in range(w):
            if hists[0,i]>=1500 and p==0:
                lanes.append(i)
                p=255
            elif hists[0,i]==0 and p==255:
                lanes.append(i)
                p=0
        if len(lanes)%2==1:
            lanes.append(w-1)
        for i in range(int(len(lanes)/2)):
            if abs(lanes[2*i]-lanes[2*i+1])>370 and t>30:
                self.stopline = True

        # get lane marking delimiters
        lanes=[]
        p=0
        for i in range(w):
            if hist[0,i]>=1500 and p==0:
                lanes.append(i)
                p=255
            elif hist[0,i]==0 and p==255:
                lanes.append(i)
                p=0
        if len(lanes)%2==1:
            lanes.append(w-1)

        # get lane markings
        centers=[]
        for i in range(int(len(lanes)/2)):
            if abs(lanes[2*i]-lanes[2*i+1])>350 and t>50:
                self.stopline = True
            elif abs(lanes[2*i]-lanes[2*i+1])>3: # and abs(lanes[2*i]-lanes[2*i+1])<100: #exclude large lanes
                centers.append((lanes[2*i]+lanes[2*i+1])/2)
        
        # get lane centers based on 4 cases
        if len(centers)==0: # no lane detected
            return w/2
        elif len(centers)==1: # one lane detected
            if centers[0]>w/2:
                return (centers[0]-0)/2
            else:
                return (centers[0]*2+640)/2
        elif abs(centers[0]-centers[len(centers)-1])<200: # the left most lane and the right most lane are close together (fuse them)
            if (centers[0]+centers[len(centers)-1])>w:
                return ((centers[0]+centers[len(centers)-1])/2+0)/2
            else:
                return ((centers[0]+centers[len(centers)-1])+640)/2
        else: # the left most lane and the right most lane are far (avg them)
            return (centers[0]+centers[len(centers)-1])/2

if __name__ == '__main__':
    try:
        node = LaneDetector()
        node.rate.sleep()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()