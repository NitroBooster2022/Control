#!/usr/bin/env python3

import sys
print("sys: ", sys.path)
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

cv_image = cv2.imread('/home/scandy/Documents/BFMC_pkgs/Control/scripts/a.jpg')
cv_image = cv2.resize(cv_image, (640,480))
print("shape: ", cv_image.shape)
def image_pub():
    rospy.init_node("image_pub")
    image_pub = rospy.Publisher('camera/color/image_raw', Image, queue_size =10)

    bridge = CvBridge()
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding = "bgr8")
        print("sent")
        image_pub.publish(image_message)
        rate.sleep()

if __name__ == "__main__":
    image_pub()