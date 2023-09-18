#! /usr/bin/env python3

import rospy
import numpy as np
from utils.msg import localisation, IMU, encoder, odometry

import math
import matplotlib.pyplot as plt
from gazebo_msgs.msg import ModelStates

import message_filters
from message_filters import ApproximateTimeSynchronizer
import argparse
from std_msgs.msg import Float32, Header
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf

class Odom():
    #initialization
    def __init__(self, simulation = False):
        rospy.init_node('control_node', anonymous=True)
        self.rateVal = 5.0
        self.rate = rospy.Rate(self.rateVal)
        self.wheelbase = 0.27
        self.header = Header()
        # Create a TransformBroadcaster object
        self.br = tf2_ros.TransformBroadcaster()
        # Create a TransformStamped object to store the transform
        self.transform_stamped = TransformStamped()
        self.useOdom = True

        #simulation
        self.simulation = simulation
        if self.simulation:
            print("Simulation mode")
            self.odomRatio = 1
            self.process_yaw = self.process_yaw_sim
        else:
            # get initial yaw from IMU
            self.initialYaw = 0
            #launch sensors at 0 to remove this
            while self.initialYaw==0:
                imu = rospy.wait_for_message("/automobile/IMU",IMU)
                self.initialYaw = imu.yaw
                print("initialYaw: "+str(self.initialYaw))
            print("Real mode")
            self.odomRatio = 0.0066
            self.process_yaw = self.process_yaw_real

        self.yaw = 1.5707
        self.velocity = 0.0
        self.odomX = 0
        self.odomY = 0
        self.odomYaw = 0
        self.gps_x = 0.0
        self.gps_y = 0.0
        self.steer = 0.0
        self.initializationFlag = False
        self.initializationTimer = None

        #timers
        self.timerodom = rospy.Time.now()

        # Publishers
        # self.odom_pub = rospy.Publisher("/automobile/odometry", odometry, queue_size=10)
        # self.odom_msg = odometry()

        # Subscribe to topics
        self.localization_sub = message_filters.Subscriber("/automobile/localisation", localisation, queue_size=3)
        self.imu_sub = message_filters.Subscriber("/automobile/IMU", IMU, queue_size=3)
        self.encoder_sub = message_filters.Subscriber("/automobile/encoder", encoder, queue_size=3)
        # self.steering_sub = message_filters.Subscriber("/automobile/steering", Float32, queue_size=3)
        self.steering_sub = rospy.Subscriber("/automobile/steering", Float32,callback=self.steer_callback, queue_size=3)
        self.model_sub = message_filters.Subscriber("/gazebo/model_states", ModelStates, queue_size=3)

        self.subscribers = []
        # self.subscribers.append(self.twist_sub)
        self.subscribers.append(self.localization_sub)
        self.subscribers.append(self.imu_sub)
        self.subscribers.append(self.encoder_sub)
        self.subscribers.append(self.model_sub)
        
        # Create an instance of TimeSynchronizer
        ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=3, slop=0.0015)
        ts.registerCallback(self.callback)

    def process_yaw_sim(self, yaw):
        self.yaw = yaw if yaw>0 else (6.2831853+yaw)
    def process_yaw_real(self, yaw):
        if yaw!=0:
            newYaw = -((yaw-self.initialYaw)*3.14159/180)
            self.yaw = newYaw if newYaw>0 else (6.2831853+newYaw)
    def steer_callback(self, steering):
        self.steer = steering.data
    def callback(self, localization, imu, encoder, model):
        self.gps_x = localization.posA
        self.gps_y = 15.0-localization.posB
        self.velocity = encoder.speed
        self.process_yaw(imu.yaw)
        # try:
        #     i = ModelStates.name.index("automobile") # index of the car
        # except:
        #     return
        # self.gps_x = ModelStates.pose[i].position.x
        # self.gps_y = ModelStates.pose[i].position.y
        if self.initializationTimer is None:
            self.initializationTimer = rospy.Time.now() + rospy.Duration(3.57)
        elif rospy.Time.now() < self.initializationTimer:
            self.odomYaw = self.yaw
            print(f"intializing... gps_x: {self.gps_x:.2f}, gps_y: {self.gps_y:.2f}")
            self.odomX = self.gps_x
            self.odomY = self.gps_y
            print(f"odomX: {self.odomX:.2f}, odomY: {self.odomY:.2f}")
            return

        self.update_states_rk4(self.velocity, self.steer)

        #print the time in seconds with 2 decimal places and the odom values
        print(f"time: {rospy.Time.now().to_sec():.2f}, odomx: {self.odomX:.2f}, odomy: {self.odomY:.2f}, gps_x: {self.gps_x:.2f}, gps_y: {self.gps_y:.2f}, x_error: {abs(self.odomX-self.gps_x):.2f}, y_error: {abs(self.odomY-self.gps_y):.2f}, yaw: {self.odomYaw:.2f}, steer: {self.steer:.2f}, speed: {self.velocity:.2f}")

        # Publish odometry message
        # self.header.stamp = rospy.Time.now()
        # self.odom_pub.header = self.header
        # self.odom_msg.odomX = self.odomX
        # self.odom_msg.odomY = self.odomY
        # self.odom_msg.odomYaw = self.odomYaw
        # self.odom_pub.publish(self.odom_msg)

        if not self.useOdom:
            try:
                i = model.name.index("automobile") # index of the car
            except:
                print("car not found")
                return
            
            # Set translation and rotation 
            self.transform_stamped.transform.translation.x = model.pose[i].position.x
            self.transform_stamped.transform.translation.y = model.pose[i].position.y
            self.transform_stamped.transform.translation.z = model.pose[i].position.z
            
            self.transform_stamped.transform.rotation.x = model.pose[i].orientation.x
            self.transform_stamped.transform.rotation.y = model.pose[i].orientation.y
            self.transform_stamped.transform.rotation.z = model.pose[i].orientation.z
            self.transform_stamped.transform.rotation.w = model.pose[i].orientation.w
        else: 
            self.transform_stamped.transform.translation.x = self.odomX
            self.transform_stamped.transform.translation.y = self.odomY
            self.transform_stamped.transform.translation.z = 0.0
            
            yaw = self.odomYaw
            qtn = tf.transformations.quaternion_from_euler(0, 0, yaw)
            # these are in quaternion
            self.transform_stamped.transform.rotation.x = qtn[0]
            self.transform_stamped.transform.rotation.y = qtn[1]
            self.transform_stamped.transform.rotation.z = qtn[2]
            self.transform_stamped.transform.rotation.w = qtn[3]
        # Broadcast the transform
        self.br.sendTransform(self.transform_stamped)

    def update_states_rk4(self, speed, steering_angle):
        # dead reckoning odometry using Runge-Kutta 4th order method
        dt = (rospy.Time.now()-self.timerodom).to_sec()
        self.timerodom = rospy.Time.now()
        magnitude = speed * dt * self.odomRatio
        yaw_rate = magnitude * math.tan(-steering_angle*math.pi/180) / self.wheelbase
        
        k1_x = magnitude * math.cos(self.yaw)
        k1_y = magnitude * math.sin(self.yaw)
        k1_yaw = yaw_rate

        k2_x = magnitude * math.cos(self.yaw + dt/2 * k1_yaw)
        k2_y = magnitude * math.sin(self.yaw + dt/2 * k1_yaw)
        k2_yaw = yaw_rate

        k3_x = magnitude * math.cos(self.yaw + dt/2 * k2_yaw)
        k3_y = magnitude * math.sin(self.yaw + dt/2 * k2_yaw)
        k3_yaw = yaw_rate

        k4_x = magnitude * math.cos(self.yaw + dt * k3_yaw)
        k4_y = magnitude * math.sin(self.yaw + dt * k3_yaw)
        k4_yaw = yaw_rate

        self.odomX += 1 / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        self.odomY += 1 / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        self.odomYaw += 1 / 6 * (k1_yaw + 2 * k2_yaw + 2 * k3_yaw + k4_yaw)

        self.odomYaw = np.fmod(self.odomYaw, 2*math.pi)
        if self.odomYaw < 0: #convert to 0-2pi
            self.odomYaw += 2*np.pi
        return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Odometry Node for Robot Control.')
    parser.add_argument("--simulation", type=str, default=True, help="Run the robot in simulation or real life")
    args = parser.parse_args(rospy.myargv()[1:])
    s = args.simulation=="True"
    node = Odom(simulation=True)
    while not rospy.is_shutdown():
        rospy.spin()
        node.rate.sleep()