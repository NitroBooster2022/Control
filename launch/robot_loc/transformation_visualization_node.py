#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs
import tf2_ros

from nav_msgs.msg import Odometry

def callback(data):
  rospy.loginfo('data: %s',data)

if __name__ == '__main__':
  rospy.init_node('tf_visualisation', anonymous=True)
  rate = rospy.Rate(5)
  rospy.loginfo('start')
  print('start')
  sub = rospy.Subscriber("odometry/filtered_imu", Odometry, callback)
  rate.sleep()
  rospy.spin()

  # tfBuffer = tf2_ros.Buffer()
  # listener = tf2_ros.TransformListener(tfBuffer)

  # rate = rospy.Rate(2)
  # while not rospy.is_shutdown():
  #   try:
  #     base_link_to_map_transform = tfBuffer.lookup_transform("map", "base_link", rospy.Time())
  #   except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
  #     rate.sleep()
  #     continue
  #   rospy.loginfo(base_link_to_map_transform)
  #   rate.sleep()