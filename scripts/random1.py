#!/usr/bin/env python3
import rospy
import numpy as np
from message_filters import ApproximateTimeSynchronizer
from std_msgs.msg import String, Byte
from utils.msg import Lane, Sign, localisation, IMU, encoder
import math

import cv2
import os
import json
import threading
import argparse

class StateMachine():
    #initialization
    def __init__(self, simulation = True, planned_path = "/paths/path.json", custom_path = False):
        rospy.init_node('lane_follower_node', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher("/automobile/command", String, queue_size=3)
        self.rate = rospy.Rate(50)
        self.lane_sub = rospy.Subscriber('lane', Lane, self.lane_callback, queue_size=3)
        self.sign_sub = rospy.Subscriber('sign', Sign, self.sign_callback, queue_size=3)
        self.imu_sub = rospy.Subscriber("/automobile/IMU", IMU, self.imu_callback, queue_size=3)
        self.encoder_sub = rospy.Subscriber("/automobile/encoder", encoder, self.encoder_callback, queue_size=3)
        self.lock = threading.Lock()

        self.callback_thread = threading.Thread(target=self.run_callback)
        self.action_thread = threading.Thread(target=self.run_action)
        self.callback_thread.start()
        self.action_thread.start()

    def run_callback(self):
        while not rospy.is_shutdown():
            rospy.spin()
    def run_action(self):
        while not rospy.is_shutdown():
            act = self.action()
            if int(act)==1:
                print(f"-----transitioning to '{self.states[self.state]}'-----")
                if self.state==0:
                    print("Speed is at "+str(self.maxspeed)+"m/s")
            self.rate.sleep()
    #callback function
    def lane_callback(self,lane):
        self.lock.acquire()
        self.center = lane.center
        self.ArrivedAtStopline = lane.stopline
        # if there's a big shift in lane center: ignore due to delay
        if abs(self.center-self.pl)>250:
            self.center = self.pl
        # ignore one center measurement when we don't detect
        if self.center==320:
            c = self.center
            self.center = self.pl
            self.pl = c
        else:
            self.pl = self.center
        self.lock.release()
    def sign_callback(self,sign):
        self.lock.acquire()
        self.detected_objects = sign.objects
        self.numObj = sign.num
        self.box1 = sign.box1
        self.box2 = sign.box2
        self.box3 = sign.box3
        self.confidence = sign.confidence
        self.lock.release()
    def encoder_callback(self,encoder):
        self.lock.acquire()
        self.velocity = encoder.speed
        self.lock.release()
    def imu_callback(self,imu):
        self.lock.acquire()
        self.process_yaw(imu.yaw)
        self.lock.release()

    #state machine
    def action(self):
        if self.state==0: #lane following
            return self.lanefollow()
        elif self.state == 1: #Approaching Intersection
            return self.approachInt()
if __name__ == '__main__':
    node = StateMachine()
    node.callback_thread.join()
    node.action_thread.join()