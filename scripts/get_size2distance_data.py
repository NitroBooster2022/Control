#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from message_filters import Subscriber, ApproximateTimeSynchronizer, TimeSynchronizer
from utils.msg import Lane, Sign, localisation, IMU, Sensors, encoder
import message_filters
import numpy as np
# Callback function for synchronized messages
def sync_callback(sign_msg, localization_msg):
    global distances, sizes
    height = -1
    detected_objects = sign_msg.objects
    numObj = sign_msg.num
    confidence = sign_msg.confidence
    boxes = [sign_msg.box1, sign_msg.box2, sign_msg.box3, sign_msg.box4]
    for i in range(numObj):
        if detected_objects[i] == 4 and confidence[i] > 0.7:
            height = boxes[i][3]
            break
    if height > 0:
        sizes.append(height)
        distance = localization_msg.posA
        distances.append(distance)
        rospy.loginfo(f"Size: {height}, Distance: {distance}")
def shutdown():
    global distances, sizes
    np.save("distances", np.array(distances))
    np.save("sizes", np.array(sizes))
    print("distances: ", distances)
    print("sizes: ", sizes)
# Initialize node and subscribers
rospy.init_node("synchronized_subscriber", anonymous=True)
sign_sub = Subscriber("sign", Sign, queue_size=3)
localization_sub = Subscriber("/automobile/localisation", localisation, queue_size=3)

# Synchronize messages using ApproximateTimeSynchronizer
sync = ApproximateTimeSynchronizer([sign_sub, localization_sub], queue_size=10, slop=0.1)
sync.registerCallback(sync_callback)

# Initialize distances and sizes arrays
distances = []
sizes = []
rate = rospy.Rate(20)
rospy.loginfo("Waiting for messages...")

while not rospy.is_shutdown():
    rospy.on_shutdown(shutdown)
    rospy.spin()
    rate.sleep()
