#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from utils.msg import localisation


class gps():
    def __init__(self):
        rospy.init_node('gps converter', anonymous=True)
        self.rate = rospy.Rate(5)
        self.pub = rospy.Publisher('localisation/gps', PoseWithCovarianceStamped, queue_size=10)
        self.sub = rospy.Subscriber("/automobile/localisation", localisation, self.callback)
        self.rate.sleep()
        rospy.spin()
    
    def callback(self, data):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'odom'
        msg.header.stamp.secs = int(rospy.get_time())
        msg.pose.pose.position.x = data.posA
        msg.pose.pose.position.y = data.posB
        self.pub.publish(msg)
        # rospy.loginfo(msg)

if __name__ == '__main__':
    try:
        gps()
    except rospy.ROSInterruptException:
        pass