#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from utils.msg import IMU
import math


class imu():
    def __init__(self):
        rospy.init_node('imu converter', anonymous=True)
        self.rate = rospy.Rate(5)
        self.pub = rospy.Publisher('localisation/imu', Imu, queue_size=10)
        self.sub = rospy.Subscriber("/automobile/IMU", IMU, self.callback)
        self.rate.sleep()
        rospy.spin()
    
    def callback(self, data):
        msg = Imu()
        msg.header.frame_id = 'base_link'
        msg.header.stamp.secs = int(rospy.get_time())
        accel = Vector3()
        accel.x = data.accelx
        accel.y = data.accely
        accel.z = data.accelz
        orient = Quaternion()
        q = self.quaternion_from_euler(data.roll,data.pitch,data.yaw)
        orient.w = q[0]
        orient.x = q[1]
        orient.y = q[2]
        orient.z = q[3]
        msg.linear_acceleration = accel
        msg.orientation = orient
        self.pub.publish(msg)
        # rospy.loginfo(msg)

    def quaternion_from_euler(self, roll, pitch, yaw):
        """
        Converts euler roll, pitch, yaw to quaternion (w in last place)
        quat = [w, x, y, z]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr

        return q



if __name__ == '__main__':
    try:
        imu()
    except rospy.ROSInterruptException:
        pass