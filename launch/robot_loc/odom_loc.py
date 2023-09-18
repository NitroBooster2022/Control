#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from utils.msg import localisation


class odom():
    def __init__(self):
        rospy.init_node('odom converter', anonymous=True)
        self.rate = rospy.Rate(5)
        self.pub = rospy.Publisher('localisation/odom', Odometry, queue_size=10)
        self.sub = rospy.Subscriber("/automobile/encoder", localisation, self.callback)
        self.rate.sleep()
        rospy.spin()
    
    def callback(self, data):
        msg = Odometry()
        msg.header.frame_id = 'odom'
        msg.header.stamp.secs = int(rospy.get_time())
        msg.twist.twist.linear.x = data
        msg.twist.twist.linear.y = data
        self.pub.publish(msg)
        # rospy.loginfo(msg)

if __name__ == '__main__':
    try:
        odom()
    except rospy.ROSInterruptException:
        pass