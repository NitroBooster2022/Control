#!/usr/bin/env python3

import rospy
import numpy as alex
from std_msgs.msg import Header, Float32
from message_filters import ApproximateTimeSynchronizer
from gazebo_msgs.msg import ModelStates, LinkStates
import math
from utils.msg import encoder, IMU
import message_filters
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion

class EncoderNode():
    def __init__(self):
        print("init encoder node")
        rospy.init_node('encoder_node', anonymous=True)
        self.model_sub = message_filters.Subscriber("/gazebo/model_states", ModelStates, queue_size=3)
        self.link_sub = message_filters.Subscriber("/gazebo/link_states", LinkStates, queue_size=3)
        self.imu_sub = rospy.Subscriber("/automobile/IMU", IMU, self.imu_callback, queue_size=3)
        self.pub = rospy.Publisher("/automobile/encoder", encoder, queue_size = 3)
        self.steer_pub = rospy.Publisher("/automobile/steering", Float32, queue_size = 3)
        # self.steer_sub = rospy.Subscriber("/automobile/steering", Float32, self.steer_callback, queue_size = 3)

        self.subscribers = [self.model_sub, self.link_sub]
        self.ts = ApproximateTimeSynchronizer(self.subscribers, 10, slop = 0.00033, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        self.p = encoder()
        self.rate = rospy.Rate(15)
        self.yaw = 0
        self.left_pose = None
        self.right_pose = None
        self.right_inertial = None
        self.left_inertial = None
        self.car_pose = None
        self.car_inertial = None
        self.car_index = None
        self.left_index = None
        self.right_index = None
        self.steer = 0
        self.angleCount = 0
        self.br = tf2_ros.TransformBroadcaster()
        # Set up a timer to run the publisher loop at 15 Hz
        self.timer = rospy.Timer(rospy.Duration(1.0 / 15.0), self.publish_encoder)

    def steer_callback(self, msg):
        self.steer = msg.data
    def callback(self, model, link):
        self.model_callback(model)
        self.link_callback(link)
    def model_callback(self, model):

        if self.car_index is None:
            try:
                self.car_idx = model.name.index("automobile")
            except ValueError:
                return

        self.car_pose = model.pose[self.car_idx]
        self.car_inertial = model.twist[self.car_idx]

    def link_callback(self, link):
        if self.left_index is None or self.right_index is None:
            try:
                left_idx = link.name.index("automobile::wheel_front_left::link_rim")
                right_idx = link.name.index("automobile::wheel_front_right::link_rim")
            except ValueError:
                return

        self.left_pose = link.pose[left_idx]
        self.right_pose = link.pose[right_idx]
        self.right_inertial = link.twist[right_idx]
        self.left_inertial = link.twist[left_idx]
        # print("left: ", self.left_pose, self.left_inertial)
        # print("right: ", self.right_pose, self.right_inertial)

    def imu_callback(self, imu):
        self.yaw = imu.yaw #between pi to -pi

    def publish_encoder(self, event):
        header = Header()
        header.frame_id = 'encoder'
        header.stamp = rospy.Time.now()
        self.p.header = header

        x_speed = self.car_inertial.linear.x
        y_speed = self.car_inertial.linear.y
        syaw = math.atan2(y_speed, x_speed)
        speed = math.sqrt(x_speed**2 + y_speed**2)
        # print("car speed: ", speed)
        error = abs(syaw - self.yaw)
        if error >= 5.73:
            error -= 6.28
        if abs(error) < 1.57:
            self.p.speed = speed
        else:
            self.p.speed = -speed

        self.pub.publish(self.p)


        # tf
        dynamic_transforms = []

        # Compute the relative position of the car with respect to the world

        tCar = TransformStamped()
        tCar.header.stamp = rospy.Time.now()
        tCar.header.frame_id = "odom"
        tCar.child_frame_id = "chassis"
        tCar.transform.translation.x = self.car_pose.position.x
        tCar.transform.translation.y = self.car_pose.position.y
        tCar.transform.translation.z = self.car_pose.position.z
        tCar.transform.rotation.x = self.car_pose.orientation.x
        tCar.transform.rotation.y = self.car_pose.orientation.y
        tCar.transform.rotation.z = self.car_pose.orientation.z
        tCar.transform.rotation.w = self.car_pose.orientation.w

        dynamic_transforms.append(tCar)

        # Compute the relative position of the left wheel with respect to the chassis
        rel_x = self.left_pose.position.x - self.car_pose.position.x
        rel_y = self.left_pose.position.y - self.car_pose.position.y
        rel_z = self.left_pose.position.z - self.car_pose.position.z

        # Compute the relative orientation of the left wheel with respect to the chassis
        # This assumes that the orientations are given as quaternions in the form [x, y, z, w]
        inv_car_orientation = quaternion_inverse([self.car_pose.orientation.x, self.car_pose.orientation.y, self.car_pose.orientation.z, self.car_pose.orientation.w])
        rel_orientation = quaternion_multiply(inv_car_orientation, [self.left_pose.orientation.x, self.left_pose.orientation.y, self.left_pose.orientation.z, self.left_pose.orientation.w])

        # Populate the transform message
        tLeft = TransformStamped()
        tLeft.header.stamp = rospy.Time.now()
        tLeft.header.frame_id = "chassis"
        tLeft.child_frame_id = "wheel_front_left"
        tLeft.transform.translation.x = rel_x
        tLeft.transform.translation.y = rel_y
        tLeft.transform.translation.z = rel_z
        tLeft.transform.rotation.x = rel_orientation[0]
        tLeft.transform.rotation.y = rel_orientation[1]
        tLeft.transform.rotation.z = rel_orientation[2]
        tLeft.transform.rotation.w = rel_orientation[3]

        dynamic_transforms.append(tLeft)

        # Compute the relative position of the right wheel with respect to the chassis
        rel_x = self.right_pose.position.x - self.car_pose.position.x
        rel_y = self.right_pose.position.y - self.car_pose.position.y
        rel_z = self.right_pose.position.z - self.car_pose.position.z

        # Compute the relative orientation of the right wheel with respect to the chassis
        # This assumes that the orientations are given as quaternions in the form [x, y, z, w]
        inv_car_orientation = quaternion_inverse([self.car_pose.orientation.x, self.car_pose.orientation.y, self.car_pose.orientation.z, self.car_pose.orientation.w])
        rel_orientation = quaternion_multiply(inv_car_orientation, [self.right_pose.orientation.x, self.right_pose.orientation.y, self.right_pose.orientation.z, self.right_pose.orientation.w])

        # Populate the transform message
        tRight = TransformStamped()
        tRight.header.stamp = rospy.Time.now()
        tRight.header.frame_id = "chassis"
        tRight.child_frame_id = "wheel_front_right"
        tRight.transform.translation.x = rel_x
        tRight.transform.translation.y = rel_y
        tRight.transform.translation.z = rel_z
        tRight.transform.rotation.x = rel_orientation[0]
        tRight.transform.rotation.y = rel_orientation[1]
        tRight.transform.rotation.z = rel_orientation[2]
        tRight.transform.rotation.w = rel_orientation[3]

        dynamic_transforms.append(tRight)
        self.br.sendTransform(dynamic_transforms)

        # publish the steering of the wheel relative to the car
        self.publish_steering_angle()

    def publish_steering_angle(self):
        # Convert the orientation quaternion of the car to Euler angles
        euler_angles_car = euler_from_quaternion([self.car_pose.orientation.x, self.car_pose.orientation.y, self.car_pose.orientation.z, self.car_pose.orientation.w])
        yaw_car = euler_angles_car[2]  # Yaw component of the car
        
        # Convert the orientation quaternion of the left wheel to Euler angles
        euler_angles_wheel = euler_from_quaternion([self.left_pose.orientation.x, self.left_pose.orientation.y, self.left_pose.orientation.z, self.left_pose.orientation.w])
        yaw_wheel = euler_angles_wheel[2]  # Yaw component of the left wheel
        
        # Compute the steering angle as the difference in yaw between the wheel and the car
        steering_angle = -(yaw_wheel - yaw_car)*180/math.pi
        # change to -pi/2 to pi/2
        if steering_angle > 90:
            steering_angle -= 180
        elif steering_angle < -90:
            steering_angle += 180
        if abs(steering_angle) > 23:
            self.angleCount += 1
            steering_angle = 23.0 * steering_angle / abs(steering_angle)
        print("real steering angle: ", round(steering_angle, 2), ", command: ", round(self.steer, 2), ", error: ", round(steering_angle - self.steer, 2), ", angleCount: ", self.angleCount)
        # Publish the steering angle
        self.steer_pub.publish(Float32(steering_angle))


if __name__ == '__main__':
    import rospy
    while not rospy.is_shutdown():
        try:
            node = EncoderNode()
            node.rate.sleep()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass