#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelStates
from utils.msg import odometry
import tf
class Node():
    def __init__(self):
        rospy.init_node('tf_dynamic_node', anonymous=True)
        self.useOdom = True
        if not self.useOdom:
            self.twist_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback, queue_size=3)
        else:
            self.odom_sub = rospy.Subscriber("/automobile/odometry", odometry, self.callback, queue_size=3)
        self.rate = rospy.Rate(15)
        # Create a TransformBroadcaster object
        self.br = tf2_ros.TransformBroadcaster()
        
        # Create a TransformStamped object to store the transform
        self.transform_stamped = TransformStamped()

    def callback(self, data):
        # Populate the TransformStamped fields
        self.transform_stamped.header.stamp = rospy.Time.now()
        self.transform_stamped.header.frame_id = "odom"
        self.transform_stamped.child_frame_id = "chassis"
        if not self.useOdom:
            try:
                i = data.name.index("automobile") # index of the car
            except:
                print("car not found")
                return
            
            # Set translation and rotation 
            self.transform_stamped.transform.translation.x = data.pose[i].position.x
            self.transform_stamped.transform.translation.y = data.pose[i].position.y
            self.transform_stamped.transform.translation.z = data.pose[i].position.z
            
            self.transform_stamped.transform.rotation.x = data.pose[i].orientation.x
            self.transform_stamped.transform.rotation.y = data.pose[i].orientation.y
            self.transform_stamped.transform.rotation.z = data.pose[i].orientation.z
            self.transform_stamped.transform.rotation.w = data.pose[i].orientation.w

            # print("x: ", ModelStates.pose[i].position.x, ", y: ", ModelStates.pose[i].position.y)
        else:
            self.transform_stamped.transform.translation.x = data.odomX
            self.transform_stamped.transform.translation.y = data.odomY
            self.transform_stamped.transform.translation.z = 0.0
            
            yaw = data.odomYaw
            qtn = tf.transformations.quaternion_from_euler(0, 0, yaw)
            # these are in quaternion
            self.transform_stamped.transform.rotation.x = qtn[0]
            self.transform_stamped.transform.rotation.y = qtn[1]
            self.transform_stamped.transform.rotation.z = qtn[2]
            self.transform_stamped.transform.rotation.w = qtn[3]
        # Broadcast the transform
        self.br.sendTransform(self.transform_stamped)
        

if __name__ == '__main__':
    while not rospy.is_shutdown():
        try:
            node = Node()
            node.rate.sleep()
            rospy.spin()
        except rospy.ROSInterruptException:
            continue

