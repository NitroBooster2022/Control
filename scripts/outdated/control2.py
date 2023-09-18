#!/usr/bin/env python3
import rospy
import numpy as np
from message_filters import ApproximateTimeSynchronizer
from std_msgs.msg import String, Byte
from utils.msg import Lane, Sign, localisation, IMU, encoder
from utils.srv import get_direction, dotted, nav
import message_filters
import time
import math

import cv2
import os
import json
import threading
import argparse

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from trackmap import track_map

class StateMachine():
    #initialization
    def __init__(self, simulation = True, planned_path = "/paths/path.json", custom_path = False):
        #simulation
        self.simulation = simulation
        if self.simulation:
            print("Simulation mode")
            self.odomRatio = 1
            self.process_yaw = self.process_yaw_sim
            self.left_trajectory = self.left_trajectory_sim
            self.right_trajectory = self.right_trajectory_sim
            self.right_exit_trajectory = self.right_exit_trajectory_sim
            self.left_exit_trajectory = self.left_exit_trajectory_sim
            self.parallelParkAngle = 45
            self.initializationTime = 2
            self.maxspeed = 0.175
            file = open(os.path.dirname(os.path.realpath(__file__))+'/PIDSim.json', 'r')
            if custom_path:
                self.track_map = track_map()
                self.track_map.custum_path()
        else:
            # get initial yaw from IMU
            self.initialYaw = 0
            while self.initialYaw==0:
                imu = rospy.wait_for_message("/automobile/IMU",IMU)
                self.initialYaw = imu.yaw
                print("initialYaw: "+str(self.initialYaw))
            print("Real mode")
            self.odomRatio = 0.0066
            self.process_yaw = self.process_yaw_real
            self.left_trajectory = self.left_trajectory_real
            self.right_trajectory = self.right_trajectory_real
            self.right_exit_trajectory = self.right_exit_trajectory_real
            self.left_exit_trajectory = self.left_exit_trajectory_real
            self.parallelParkAngle = 35
            self.initializationTime = 10
            self.maxspeed = 0.125
            file = open(os.path.dirname(os.path.realpath(__file__))+'/PID.json', 'r')
            #enable PID and encoder at the start to get messages from automobile/encoder
            self.msg.data = '{"action":"5","activate": true}'
            self.cmd_vel_pub.publish(self.msg)
            self.cmd_vel_pub.publish(self.msg)
            self.cmd_vel_pub.publish(self.msg)
            self.msg.data = '{"action":"4","activate": true}'
            self.cmd_vel_pub.publish(self.msg)
            self.cmd_vel_pub.publish(self.msg)
            self.cmd_vel_pub.publish(self.msg)
            #0:left, 1:straight, 2:right, 3:parkF, 4:parkP, 5:exitparkL, 6:exitparkR, 7:exitparkP
            #8:enterhwLeft, 9:enterhwStright, 10:rdb, 11:exitrdbE, 12:exitrdbS, 13:exitrdbW, 14:curvedpath
            self.decisions = [2,3,6,0,4]
            self.decisionsI = 0
        #states
        self.states = ['Lane Following', "Approaching Intersection", "Stopping at Intersection", 
                       "Intersection Maneuvering", "Approaching Crosswalk", "Pedestrian", "Highway",
                       "Carblock", "Roundabout", "Parking", "Initial", "Parked", "Curvedpath"] #13 states
        self.state = 10 #initial

        #sign
        self.class_names = ['oneway', 'highwayexit', 'stopsign', 'roundabout', 'park', 'crosswalk', 'noentry', 'highwayentrance', 'priority',
                'lights','block','pedestrian','car','others','nothing']
        self.min_sizes = [25,25,22,000,45,42,25,25,25,80,100,72,130]
        self.max_sizes = [50,75,70,000,75,80,50,75,50,200,150,200,300]
        self.center = -1
        self.detected_objects = []
        self.numObj = -1
        self.box1 = []
        self.box2 = []

        #pose (will get them from localisation)
        self.x = 0.82
        self.y = 14.91
        self.yaw = 1.5707
        self.speed = 0
        self.odomX = 0
        self.odomY = 0

        #PID
        #for initial angle adjustment
        self.error_sum = 0
        self.last_error = 0
        #for goal points
        self.error_sum2 = 0
        self.last_error2 = 0
        # Load PIDs data
        data = json.load(file)
        print("PIDs params:")
        print(data)
        self.p = data.get('p')
        self.d = data.get('d')
        self.i = data.get('i')
        self.kp = data.get('kp')
        self.kd = data.get('kd')
        self.ki = data.get('ki')
        self.kp2 = data.get('kp2')
        self.kd2 = data.get('kd2')
        self.ki2 = data.get('ki2')

        #steering
        self.msg = String()
        self.msg2 = String()
        self.last = 0
        self.pl = 320 # previous lane center

        #timers
        self.timer = None
        self.timer2 = None
        self.timer3 = None

        #intersection & parking
        #0:left, 1:straight, 2:right, 3:parkF, 4:parkP, 5:exitparkL, 6:exitparkR, 7:exitparkP
        #8:enterhwLeft, 9:enterhwStright, 10:rdb, 11:exitrdbE, 12:exitrdbS, 13:exitrdbW, 14:curvedpath
        self.intersectionStop = None
        self.intersectionDecision = -1
        self.parkingDecision = -1
        self.exitDecision = -1
        self.rdbDecision = -1
        self.decisionList = ["left","straight","right","parkF","parkP",
        "exitparkL","exitparkR","exitparkP","enterhwLeft","enterhwStright","rdb",
        "exitrdbE","exitrdbS","exitrdbW","curvedpath"]
        self.doneManeuvering = False
        self.doneParking = False
        self.destination_x = None
        self.destination_y = None
        self.destination_theta = None
        self.initialPoints = None
        self.intersectionState = 0
        self.numIntersectionStates = 0
        self.trajectory = None
        #constants
        self.orientation = 1 #0,1,2,3=east,north,west,south
        self.directions = ["east", "north", "west", "south"]
        self.orientations = np.array([0,1.5708,3.14159,4.7124]) #0, pi/2, pi, 3pi/2
        self.left_offset_x = 1.20
        self.left_offset_y = 0.82
        self.right_offset_x = 0.80
        self.right_offset_y = 0.573
        self.velocity = self.maxspeed
        self.offsets_x = np.array([self.left_offset_x, self.left_offset_x*1.2, self.right_offset_x])
        self.offsets_y = np.array([self.left_offset_y, 0, self.right_offset_y])
        self.rotation_matrices = np.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]]]) #E,N,W,S

        #carblock
        self.carCleared = None
        self.dotted = False
        self.ArrivedAtStopline = False
        """
        Initialize the lane follower node
        """
        rospy.init_node('lane_follower_node', anonymous=True)
        self.timer4 = rospy.Time.now()
        self.timer5 = rospy.Time.now()
        self.timer6 = rospy.Time.now()
        self.odomTimer = rospy.Time.now()
        self.t2 = rospy.Time.now()
        self.cmd_vel_pub = rospy.Publisher("/automobile/command", String, queue_size=3)
        self.rate = rospy.Rate(50)
        self.dt = 1/50 #for PID

        # Create service proxy
        # self.get_dir = rospy.ServiceProxy('get_direction',get_direction)
        # self.get_dotted = rospy.ServiceProxy('dotted',dotted)
        # rospy.wait_for_service('dotted')
        # how to use:
        # d = self.get_dotted("dotted").dotted

        # Subscribe to topics
        self.lane_sub = message_filters.Subscriber('lane', Lane, queue_size=3)
        self.sign_sub = message_filters.Subscriber('sign', Sign, queue_size=3)
        # self.localization_sub = message_filters.Subscriber("/automobile/localisation", localisation, queue_size=3)
        self.imu_sub = message_filters.Subscriber("/automobile/IMU", IMU, queue_size=3)
        self.encoder_sub = message_filters.Subscriber("/automobile/encoder", encoder, queue_size=3)
        self.subscribers = []
        self.subscribers.append(self.lane_sub)
        self.subscribers.append(self.sign_sub)
        # self.subscribers.append(self.localization_sub)
        self.subscribers.append(self.imu_sub)
        self.subscribers.append(self.encoder_sub)
        
        # Create an instance of TimeSynchronizer
        ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=3, slop=1.15)
        ts.registerCallback(self.callback)

        self.callback_thread = threading.Thread(target=self.run_callback)
        self.action_thread = threading.Thread(target=self.run_action)
        self.callback_thread.start()
        self.action_thread.start()

        #stop at shutdown
        def shutdown():
            pub = rospy.Publisher("/automobile/command", String, queue_size=3)
            msg = String()
            msg2 = String()
            # msg.data = '{"action":"3","brake (steerAngle)":'+str(0.0)+'}'
            msg.data = '{"action":"1","speed":'+str(0.0)+'}'
            msg2.data = '{"action":"2","steerAngle":'+str(0.0)+'}'
            for haha in range(3):
                pub.publish(msg2)
                pub.publish(msg)
                self.rate.sleep()
        
        rospy.on_shutdown(shutdown)

        # timer to stop before parking
        self.timerP = None

        self.parkAdjust = True
        self.offset = 0.3
        self.parksize = 0

        self.light = False #light detected in stop intersection state
        self.hw = False
        self.rdb = False
        self.rdbExitYaw = 0
        self.rdbTransf = 0
        self.timerO = None
        self.carBlockSem = -1
        # self.trackbars()

        if self.simulation:
            imu = rospy.wait_for_message("/automobile/IMU",IMU)
            self.yaw = imu.yaw
            loc = rospy.wait_for_message("/automobile/localisation",localisation)
            self.x = loc.posA
            self.y = loc.posB
            print("x,y,yaw",self.x,self.y,self.yaw)

            # self.planned_path=['parkingN','roundabout','int5N','int1E','roundabout','parkingN']
            if not custom_path:
                self.planned_path = json.load(open(os.path.dirname(os.path.realpath(__file__))+planned_path, 'r'))
                self.track_map = track_map(self.x,self.y,self.yaw,self.planned_path)
                self.track_map.plan_path()
            else:
                self.track_map.location = self.track_map.locate(self.x,self.y,self.yaw)
                self.track_map.plan_path()
            # self.track_map.draw_map()
            #0:left, 1:straight, 2:right, 3:parkF, 4:parkP, 5:exitparkL, 6:exitparkR, 7:exitparkP
            #8:enterhwLeft, 9:enterhwStright, 10:rdb, 11:exitrdbE, 12:exitrdbS, 13:exitrdbW, 14:curvedpath
            self.decisions = self.track_map.directions
            self.decisionsI = 0
    
    def process_yaw_sim(self, yaw):
        self.yaw = yaw if yaw>0 else (6.2831853+yaw)
    def process_yaw_real(self, yaw):
        if yaw!=0:
            newYaw = -((yaw-self.initialYaw)*3.14159/180)
            self.yaw = newYaw if newYaw>0 else (6.2831853+newYaw)

    def run_callback(self):
        rospy.spin()
    def run_action(self):
        while not rospy.is_shutdown():
            print(f"time: {rospy.Time.now()- self.t2}")
            self.t2 = rospy.Time.now()
            act = self.action()
            if int(act)==1:
                print(f"-----transitioning to '{self.states[self.state]}'-----")
                if self.state==0:
                    print("Speed is at "+str(self.maxspeed)+"m/s")
            self.rate.sleep()
    #callback function
    def callback(self,lane,sign,imu,encoder):

        self.dt = (rospy.Time.now()-self.timer6).to_sec()
        self.timer6 = rospy.Time.now()

        # Perform decision making tasks

        # self.x = localization.posA
        # self.y = 15.0-localization.posB
        self.process_yaw(imu.yaw)
        self.velocity = encoder.speed
        self.center = lane.center
        self.ArrivedAtStopline = lane.stopline
        self.detected_objects = sign.objects
        self.numObj = sign.num
        self.box1 = sign.box1
        self.box2 = sign.box2

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
        # print(self.center)

        # print("x,y,yaw,velocity,center,stopline: ", self.x, self.y, self.yaw, self.velocity, self.center, self.ArrivedAtStopline)
        # for i in range(self.numObj):
        #     if i == 0:
        #         print(self.numObj)
        #         print(f"{self.class_names[self.detected_objects[i]]} detected! width, height: {self.box1[2]}, {self.box1[3]}")
        #     elif i == 1:
        #         print(f"{self.class_names[self.detected_objects[i]]} detected! width, height: {self.box2[2]}, {self.box2[3]}")
        #     else:
        #         print(f"{self.class_names[self.detected_objects[i]]} detected!")
        # if int(self.action())==1:
        #     print(f"-----transitioning to '{self.states[self.state]}'-----")
        #     if self.state==0:
        #         # self.pl = 320
        #         print("Speed is at "+str(self.maxspeed)+"m/s")
    
    #state machine
    def action(self):
        if self.state==0: #lane following
            return self.lanefollow()
        elif self.state == 1: #Approaching Intersection
            return self.approachInt()
        elif self.state == 2: #Stopping at Intersection
            return self.stopInt()
        elif self.state == 3: #Intersection Maneuvering
            return self.maneuverInt()
            # return self.maneuverIntHC()
        elif self.state == 4: #Approaching Crosswalk
            return self.approachCrosswalk()
        elif self.state == 5: #Pedestrian
            return self.stopPedestrian()
        elif self.state == 6: #Highway
            return self.highway()
        elif self.state == 7: #Carblock
            return self.carBlock()
        elif self.state == 8: #Roundabout
            return self.roundabout() #not implemented yet
        elif self.state == 9: #Parking
            if self.timerP is None:
                self.timerP = rospy.Time.now() + rospy.Duration(1.57) # stop before parking
                print("prepare to park")
            elif rospy.Time.now() >= self.timerP:
                return self.park()
            self.idle()
            return 0
        elif self.state == 10: #initialization state
            if self.timer is None:
                print("Initializing controller...")
                self.timer = rospy.Time.now() + rospy.Duration(self.initializationTime)
            if rospy.Time.now() >= self.timer:
                print("done initializing.")
                self.timer = None
                self.state = 0
                return 1
            else:
                return 0
        elif self.state == 11: #parked
            if self.decisionsI >= len(self.decisions):
                self.idle()
                self.idle()
                self.idle()
                rospy.signal_shutdown("Exit")
            else:
                if self.timerP is None:
                    self.timerP = rospy.Time.now() + rospy.Duration(1.57) # stop before parking
                    print("prepare to exit")
                elif rospy.Time.now() >= self.timerP:
                    return self.exitPark()
                self.idle()
                return 0
        elif self.state == 12: #Curvedpath
            return self.curvedpath()
    
    #actions
    def lanefollow(self):
        #transition events
        if self.ArrivedAtStopline:
            print("signless intersection detected... -> state 3")
            self.doneManeuvering = False #set to false before entering state 3
            self.state = 3
            return 1
        elif self.stop_sign_detected():
            print("stop sign detected -> state 1")
            self.intersectionStop = True
            self.state = 1
            return 1
        elif self.light_detected(): #change this to stop till light turns green
            #call service to check light color
            if self.is_green():
                print("green light detected -> state 1")
                self.intersectionStop = False
            else:
                print("red light detected -> state 1")
                self.intersectionStop = True
                self.light = True
            self.state = 1
            return 1
        elif self.crosswalk_sign_detected():
            print("crosswalk sign detected -> state 4")
            self.state = 4
            return 1
        elif self.pedestrian_appears():
            print("pedestrian appears!!! -> state 5")
            self.history = self.state
            self.state = 5
            self.timer3 = rospy.Time.now()+rospy.Duration(2.5)
            return 1
        #can only enter highway from roundabout or intersection maneuver
        # elif self.highway_entrance_detected():
        #     print("entering highway -> 6")
        #     self.state = 6
        #     return 1
        elif self.car_detected() or self.carBlockSem > 0:
            # print("Carblock -> 7")
            if self.car_detected():
                self.carBlockSem = 20
            else:
                self.carBlockSem -= 1
                if self.carBlockSem == 0:
                    self.timerO = None
                    return 0
            if self.timerO == None:
                self.timerO = rospy.Time.now() + rospy.Duration(1.57)
                print("prepare to overtake")
            elif rospy.Time.now() >= self.timerO:
                self.timerO = None
                self.carBlockSem = -1
                self.history = self.state
                self.state = 7
                return 1
            else:
                self.idle()
                return 0
        elif self.entering_roundabout(): #revamp this
                self.rdb = True
                self.state = 1 #should be approaching roundabout state similar to approachInt
                return 1
        elif self.parking_detected():
            # if not at parking decision yet pass
            if self.decisionsI >= len(self.decisions):
                self.publish_cmd_vel(self.get_steering_angle())
                return 0
            elif (self.decisions[self.decisionsI] != 3 and self.decisions[self.decisionsI] != 4):
                self.publish_cmd_vel(self.get_steering_angle())
                return 0
            if self.detected_objects[0] == 4:
                self.parksize = max(self.box1[2], self.box1[3])
            else:
                self.parksize = max(self.box2[2], self.box2[3])
            self.parksize = self.parksize*0.00263
            print("about to park -> 9")
            self.state = 9
            return 1
        
        # elif self.object_detected(10): #check for reimplementation
        #     print("Block!!! -> 7")
        #     self.state = 7
        #     return 1

        # Determine the steering angle based on the center and publish the steering command
        self.publish_cmd_vel(self.get_steering_angle())
        return 0
    
    def approachInt(self):
        #Transition events
        if self.ArrivedAtStopline:
            if self.intersectionStop:
                print("arrived at stopline. Transitioning to 'stopping at intersection'.")
                self.state = 2
                return 1
            else:
                print("arrived at stopline. Transitioning to 'intersection maneuvering' directly.")
                self.doneManeuvering = False #set to false before entering state 3
                self.state = 3
                return 1
        # Determine the steering angle based on the center publish the steering command
        self.publish_cmd_vel(self.get_steering_angle()) 
        return 0
    
    def stopInt(self):
        #Transition events
        if self.timer is None:
            self.timer = rospy.Time.now() + rospy.Duration(3.57)
        elif rospy.Time.now() >= self.timer:
            self.timer = None
            self.doneManeuvering = False #set to false before entering state 3
            self.state = 3
            return 1
        elif self.light:
            #call service to check light color
            if self.is_green():
                # print("green")
                self.light = False
                self.timer = None
                self.doneManeuvering = False #set to false before entering state 3
                self.state = 3
                return 1
            else:
                # print("red")
                self.timer = rospy.Time.now() + rospy.Duration(3.57)
        self.idle()
        return 0
    
    def maneuverInt(self):
        if self.doneManeuvering:
            print("done intersection maneuvering.")
            self.doneManeuvering = False #reset
            self.intersectionDecision = -1 #reset
            if self.hw:
                print("entering highway -> 6")
                self.state = 6
            else:
                self.state = 0 #go back to lane following
            self.hw = False
            self.initialPoints = None #reset initial points
            self.pl = 320
            return 1
        elif self.intersectionDecision < 0:
            if self.decisionsI >= len(self.decisions):
                self.idle()
                self.idle()
                self.idle()
                rospy.signal_shutdown("Exit")
            self.intersectionDecision = self.decisions[self.decisionsI] #replace this with service call
            self.decisionsI+=1
            if self.intersectionDecision == 8:
                self.intersectionDecision = 0
                self.hw = True
            elif self.intersectionDecision == 9:
                self.intersectionDecision = 1
                self.hw = True
            elif self.intersectionDecision == 10:
                print("entering roundabout -> 8")
                self.rdb = False #check this
                self.intersectionDecision = -1 #reset
                self.state = 8
                return 1
            if self.intersectionDecision == 0: #left
                self.trajectory = self.left_trajectory
            elif self.intersectionDecision == 1: #straight
                self.trajectory = self.straight_trajectory
            elif self.intersectionDecision == 2: #right
                self.trajectory = self.right_trajectory
            else:
                raise ValueError("self.intersectionDecision id wrong: ",self.intersectionDecision)
            print("intersection decision: going " + self.decisionList[self.intersectionDecision])
        if self.initialPoints is None:
            self.set_current_angle()
            # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
            # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
            self.initialPoints = np.array([self.x, self.y])
            # print("initialPoints points: ", self.initialPoints)
            self.odomX, self.odomY = 0.0, 0.0 #reset x,y
            self.odomTimer = rospy.Time.now()
            self.intersectionState = 1 if self.intersectionDecision!=1 else 0#adjusting angle:0, trajectory following:1, adjusting angle2: 2..
        self.odometry()
        poses = np.array([self.odomX,self.odomY])
        poses = poses.dot(self.rotation_matrices[self.orientation])
        x,y = poses[0], poses[1]
        # print("position: ",x,y)
        if self.intersectionState==0: #adjusting
            error = self.yaw-self.currentAngle
            if self.yaw>=5.73: #subtract 2pi to get error between -pi and pi
                error-=6.28
            # print("yaw, curAngle, error: ", self.yaw, self.currentAngle, error)
            if abs(error) <= 0.05:
                self.intersectionState+=1 #done adjusting
                print("done adjusting angle. Transitioning to trajectory following")
                self.error_sum = 0 #reset pid errors
                self.last_error = 0
                return 0
            else:
                self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
                return 0
        elif self.intersectionState==1: #trajectory following
            desiredY = self.trajectory(x)
            error = y - desiredY
            # print("x, y_error: ",x,abs(error) )
            arrived = abs(self.yaw-self.destinationAngle) <= 0.1
            if self.intersectionDecision == 1:
                arrived = arrived and abs(x)>=self.offsets_x[self.intersectionDecision]
            # print("yaw_error: ")
            # print(str(self.yaw-self.destinationAngle))
            if arrived:
                print("trajectory done.")
                self.doneManeuvering = True
                self.last_error2 = 0 #reset pid errors
                self.error_sum2 = 0
                return 0
            # steering_angle = self.pid2(error)
            # print("steering: ",steering_angle)
            # print("x, y, desiredY, angle, steer: ", x, y, desiredY, self.yaw, steering_angle*180/3.14159)
            self.publish_cmd_vel(self.pid2(error), self.maxspeed*0.9)
            return 0
    
    def approachCrosswalk(self):
        #Transition events
        if self.timer is None: #start timer. ~13 seconds to pass crosswalk
            # 13s*0.66*0.135m/s = 1.16m
            print("slowing down to "+str(0.66*self.maxspeed)+"m/s")
            t = 1.16/(0.66*self.maxspeed) # calculate time based on speed
            self.timer = rospy.Time.now() + rospy.Duration(t)
        if rospy.Time.now() >= self.timer:
            print("crosswalk passed, speed back up to "+str(self.maxspeed)+"m/s")
            self.timer = None #reset timer
            self.state = 0
            return 1
        elif self.pedestrian_appears():
            print("pedestrian appears!!! -> state 5")
            self.timer = None #reset timer
            self.history = self.state
            self.state = 5
            self.timer3 = rospy.Time.now()+rospy.Duration(2.5)
            return 1
        #Action: slow down
        # Publish the steering command
        self.publish_cmd_vel(self.get_steering_angle(), self.maxspeed*0.66) #Slower
        return 0
    
    def stopPedestrian(self):
        if not self.pedestrian_appears():
            if rospy.Time.now() > self.timer3:
                self.timer3 = None
                self.state = self.history if self.history is not None else 0
                self.history = None
                return 1
        else:
            print("pedestrian appears!!!")
            self.timer3 = rospy.Time.now()+rospy.Duration(2.5)
        #Action: idle
        self.idle()
        return 0 
    
    def highway(self):
        # if self.highway_exit_detected():
        #     if self.entering_roundabout(): #check this
        #         self.rdb = True
        #         self.state = 1 #should be approaching roundabout state similar to approachInt
        #         return 1
        #     else: #go to approachInt
        #         self.intersectionStop = False
        #         self.state = 1
        #         return 1
        if self.decisionsI < len(self.decisions):
            if self.decisions[self.decisionsI] == 14 and self.yaw >= np.pi/12: #tune this
                self.doneManeuvering = False
                self.state = 12
                return 1
        if self.ArrivedAtStopline:
            self.doneManeuvering = False #set to false before entering state 3
            self.state = 3
            return 1
        if self.car_detected() or self.carBlockSem > 0: #overtake
            if self.car_detected():
                self.carBlockSem = 20
            else:
                self.carBlockSem -= 1
                if self.carBlockSem == 0:
                    self.timerO = None
                    return 0
            if self.timerO == None:
                self.timerO = rospy.Time.now() + rospy.Duration(1.57)
                print("prepare to overtake")
            elif rospy.Time.now() >= self.timerO:
                self.carBlockSem = -1
                self.timerO = None
                self.history = self.state
                self.state = 7
                return 1
            else:
                self.idle()
                return 0
        if self.pedestrian_appears():
            print("pedestrian appears!!! -> state 5")
            self.history = self.state
            self.state = 5
            self.timer3 = rospy.Time.now()+rospy.Duration(2.5)
            return 1
        self.publish_cmd_vel(self.get_steering_angle(), self.maxspeed*1.33)
        return 0
    
    def carBlock(self):
        #/entry: checkDotted
        #action: overtake or wait
        # if self.carCleared is None:
        #     self.carCleared = False
        # if self.dotted:#use service
        #     #overtake
        # else: #wait
        #     if self.car_detected():
        #         self.idle()
        #     else:
        #         self.carCleared = True
        #     return 0
        
        if self.doneManeuvering:
            print("done overtaking. Back to lane following...")
            self.doneManeuvering = False #reset
            self.state = self.history if self.history is not None else 0
            self.history = None
            self.initialPoints = None #reset initial points
            self.timerP = None
            self.pl = 320
            return 1
        if True: #change with left right overtaking if needed
            if self.initialPoints is None:
                self.overtaking_angle = self.yaw
                # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
                # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
                self.initialPoints = np.array([self.x, self.y])
                # print("initialPoints points: ", self.initialPoints)
                self.offset = 0
                print("begin going straight for "+str(self.offset)+"m")
                self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                self.odomTimer = rospy.Time.now()
                self.intersectionState = 0 #going straight:0, trajectory following:1, adjusting angle2: 2..
            if self.intersectionState==0: #adjusting
                error = self.yaw - (self.overtaking_angle + np.pi/4)
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, error: ", self.yaw, error)
                if abs(error) <= 0.05:
                    self.intersectionState += 1
                    print("done adjusting angle!!")
                    self.timer5 = rospy.Time.now()+rospy.Duration(3) #change to odom
                self.publish_cmd_vel(-23, self.maxspeed*0.9)
                return 0
            elif self.intersectionState==1: #adjusting
                error = self.yaw - self.overtaking_angle
                if self.history == 6: #don't need to go back exactly at highway
                    error -= np.pi/8
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                if abs(error) < 0.05:
                    if self.history == 6:#go back to highway immediatly
                        print("done changinng lane!!")
                        self.doneManeuvering = True
                        self.error_sum = 0 #reset pid errors
                        self.last_error = 0
                        return 0
                    self.intersectionState += 1
                    print("done adjusting angle!!")
                self.publish_cmd_vel(23, self.maxspeed*0.9)
                return 0
            elif self.intersectionState==2: #adjusting
                error = self.yaw - (self.overtaking_angle - np.pi/4)
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, error: ", self.yaw, error)
                if abs(error) <= 0.05:
                    self.intersectionState +=1
                    print("done adjusting angle!!")
                self.publish_cmd_vel(23, self.maxspeed*0.9)
                return 0
            elif self.intersectionState==3: #adjusting
                error = self.yaw - self.overtaking_angle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, error: ", self.yaw, error)
                if abs(error) <= 0.05:
                    print("done adjusting angle!!")
                    self.doneManeuvering = True
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                self.publish_cmd_vel(-23, self.maxspeed*0.9)
                return 0
    
    def roundabout(self):
        if self.doneManeuvering:
            print("done roundabout maneuvering. Back to lane following...")
            self.doneManeuvering = False #reset
            if self.rdbDecision == 12:
                print("entering highway -> 6")
                self.state = 6
            else:
                self.state = 0 #go back to lane following
            self.rdbDecision = -1
            self.initialPoints = None #reset initial points
            self.timerP = None
            self.pl = 320
            return 1
        elif self.rdbDecision < 0:
            if self.decisionsI >= len(self.decisions):
                self.idle()
                self.idle()
                self.idle()
                rospy.signal_shutdown("Exit")
            self.rdbDecision = self.decisions[self.decisionsI] #replace this with service call
            self.decisionsI+=1
            #could place them in set_current_angle()
            if self.rdbDecision == 11: #E
                self.rdbExitYaw = np.pi/4 #change this depending on implementation
            elif self.rdbDecision == 12: #S
                self.rdbExitYaw = 7*np.pi/4 #change this depending on implementation
            elif self.rdbDecision == 13: #W
                self.rdbExitYaw = 5*np.pi/4 #change this depending on implementation
            else:
                raise ValueError("self.rdbDecision id wrong: ",self.rdbDecision)
            print("roundabout decision: going " + self.decisionList[self.rdbDecision])
            self.trajectory = self.rdb_trajectory
        if self.initialPoints is None:
            self.set_current_angle()
            # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
            # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
            self.initialPoints = np.array([self.x, self.y])
            # print("initialPoints points: ", self.initialPoints)
            self.rdbTransf = -self.orientations[self.orientation]
            self.odomX, self.odomY = 0.0, 0.0 #reset x,y
            self.odomTimer = rospy.Time.now()
            self.offset = 0.4
            self.intersectionState = 0#adjusting angle:0, trajectory following:1, adjusting angle2: 2..
        self.odometry()
        poses = np.array([self.odomX,self.odomY])
        poses = poses.dot(self.rotation_matrices[self.orientation])
        x,y = poses[0], poses[1]
        if self.intersectionState==0: #adjusting
            error = self.yaw-self.currentAngle
            if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
            if x >= self.offset:
                self.intersectionState+=1
                self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                self.odomTimer = rospy.Time.now()
                self.error_sum = 0 #reset pid errors
                self.last_error = 0
                print("done going straight. Transitioning to trajectory following")
                # print("current angle, destination: ", self.yaw, self.destinationAngle)
            self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
            return 0
        elif self.intersectionState==1:
            self.publish_cmd_vel(15, self.maxspeed*0.9)
            yaw = self.currentAngle-np.pi/4
            yaw = yaw if yaw>0 else (6.2831853+yaw)
            arrived = abs(self.yaw-yaw) <= 0.1
            if arrived:
                print("trajectory done. adjusting angle")
                self.intersectionState += 1
                return 0
            return 0
        elif self.intersectionState==2: #trajectory following
            # desiredY = self.trajectory(x,self.rdbTransf)
            # error = y - desiredY
            # print("x,y,y_error: ",x,y,error)
            # self.publish_cmd_vel(self.pid2(error), self.maxspeed*0.9)
            self.publish_cmd_vel(-0.35, self.maxspeed*0.9)
            arrived = abs(self.yaw-self.rdbExitYaw) <= 0.1
            if arrived:
                print("trajectory done. adjusting angle")
                self.intersectionState += 1
                # self.last_error2 = 0 #reset pid errors
                # self.error_sum2 = 0
                return 0
            return 0
        elif self.intersectionState == 3: #adjust angle 2
            error = self.yaw-(self.rdbExitYaw-np.pi/4)
            if self.yaw>=5.73: #subtract 2pi to get small error
                error-=6.28
            # print("yaw, destAngle, error: ", self.yaw, self.destinationAngle, error)
            if abs(error) <= 0.1:
                print("done roundabout maneuvering!!")
                self.doneManeuvering = True
                self.error_sum = 0 #reset pid errors
                self.last_error = 0
                return 0
            else:
                self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
                return 0
        return 0

    def park(self):
        if self.doneParking:
            print("done parking maneuvering. Stopping vehicle...")
            self.doneParking = False #reset
            self.state = 11 #parked
            self.parkingDecision = -1
            self.initialPoints = None #reset initial points
            self.timerP = None
            self.pl = 320
            return 1
        elif self.parkingDecision < 0:
            if self.decisionsI >= len(self.decisions):
                self.idle()
                self.idle()
                self.idle()
                rospy.signal_shutdown("Exit")
            self.parkingDecision = self.decisions[self.decisionsI] #replace this with service call
            self.decisionsI+=1
            print("parking decision: going ") #+ self.parkingDecisions[self.parkingDecision])
            if self.parkingDecision == 3: #front parking
                self.trajectory = self.right_trajectory
            elif self.parkingDecision == 4: #parallel parking
                pass
            else:
                raise ValueError("self.parkingDecision id wrong: ",self.parkingDecision)
            print("parking decision: going " + self.decisionList[self.parkingDecision])
        if self.parkingDecision == 4:
            if self.initialPoints is None:
                self.set_current_angle()
                # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
                # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
                self.initialPoints = np.array([self.x, self.y])
                # print("initialPoints points: ", self.initialPoints)
                self.offset = 0.573 if self.simulation else 1.6 + self.parksize 
                print("begin going straight for "+str(self.offset)+"m")
                self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                self.odomTimer = rospy.Time.now()
                self.intersectionState = 0 #going straight:0, trajectory following:1, adjusting angle2: 2..
            self.odometry()
            poses = np.array([self.odomX, self.odomY])
            poses = poses.dot(self.rotation_matrices[self.orientation])
            x, y = poses[0], poses[1]
            # print("position: ",x,y)
            if self.intersectionState==0: #going straight
                error = self.yaw-self.currentAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                if x >= self.offset:
                    self.intersectionState = 1
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    print("done going straight. begin adjusting angle...")
                    # print("current angle, destination: ", self.yaw, self.destinationAngle)
                self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
                return 0
            if self.intersectionState==1: #adjusting
                error = self.yaw - self.destinationAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, error: ", self.yaw, error)
                if abs(error) >= self.parallelParkAngle*np.pi/180:
                    self.intersectionState = 3 # skip adjusting 2
                    print(f"{self.parallelParkAngle} degrees...")
                    self.timer5 = rospy.Time.now()+rospy.Duration(3) #change to odom
                self.publish_cmd_vel(23, -self.maxspeed*0.9)
                return 0
            elif self.intersectionState==2: #adjusting
                if rospy.Time.now() >= self.timer5:
                    self.intersectionState = 3
                    self.timer5 = None
                    print("done going back. begin adjusting angle round2...")
                self.publish_cmd_vel(0, -self.maxspeed*0.9)
                return 0
            elif self.intersectionState==3: #adjusting
                error = self.yaw - self.destinationAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                if abs(error) < 0.05:
                    print("done")
                    self.doneParking = True
                    return 0
                self.publish_cmd_vel(-23, -self.maxspeed*0.9)
                return 0
        elif self.parkingDecision == 3:
            if self.initialPoints is None:
                self.set_current_angle()
                # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
                # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
                self.initialPoints = np.array([self.x, self.y])
                # print("initialPoints points: ", self.initialPoints)
                self.offset = 0.3 if self.simulation else 0.12 + self.parksize
                print("begin going straight for "+str(self.offset)+"m")
                self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                self.odomTimer = rospy.Time.now()
                self.intersectionState = 0 #going straight:0, trajectory following:1, adjusting angle2: 2..
            self.odometry()
            poses = np.array([self.odomX, self.odomY])
            poses = poses.dot(self.rotation_matrices[self.orientation])
            x, y = poses[0], poses[1]
            # print("position: ",x,y)
            if self.intersectionState==0: #adjusting
                if abs(x)>=self.offset:
                    self.intersectionState+=1 #done adjusting
                    self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                    self.odomTimer = rospy.Time.now()
                    print("done going straight. Transitioning to trajectory following")
                    print(f"current odom position: ({self.odomX},{self.odomY})")
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    return 0
                else:
                    self.publish_cmd_vel(self.get_steering_angle(), self.maxspeed)
                    # error = self.yaw-self.currentAngle
                    # self.publish_cmd_vel(self.pid(error), self.maxspeed)
                    # print(str(x))
                    return 0
            elif self.intersectionState==1: #trajectory following
                desiredY = self.trajectory(x)
                error = y - desiredY
                # print("x, y error: ",x,abs(error) )
                arrived = abs(self.yaw-self.destinationAngle) <= 0.3
                if arrived:# might need to change
                    print("trajectory done. adjust angle round 2")
                    self.intersectionState += 1
                    self.last_error2 = 0 #reset pid errors
                    self.error_sum2 = 0
                    return 0
                # steering_angle = self.pid2(error)
                # print("x, y, desiredY, angle: ", x, y, desiredY, steering_angle*57.29578)
                self.publish_cmd_vel(self.pid2(error), self.maxspeed*0.9)
                return 0
            elif self.intersectionState == 2: #adjust angle 2
                error = self.yaw-self.destinationAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, destAngle, error: ", self.yaw, self.destinationAngle, error)
                if abs(error) <= 0.05:
                    print("done adjusting angle!!")
                    print("adjusting position to y between 0.4-0.5")
                    self.intersectionState += 1
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    return 0
                else:
                    if abs(y)<0.4: #adjust forward
                        self.publish_cmd_vel(self.pid(error), self.maxspeed*0.75)
                        self.parkAdjust = True
                    elif abs(y)>0.5: #adjust backward
                        self.publish_cmd_vel(-self.pid(error), -self.maxspeed*0.75)
                        self.parkAdjust = False
                    elif self.parkAdjust:
                        self.publish_cmd_vel(self.pid(error), self.maxspeed*0.75)
                    else:
                        self.publish_cmd_vel(-self.pid(error), -self.maxspeed*0.75)
                    return 0
            elif self.intersectionState == 3: #adjust position
                if abs(y)<0.4:
                    self.publish_cmd_vel(0, self.maxspeed*0.75)
                    return 0
                elif abs(y)>0.5:
                    self.publish_cmd_vel(0, -self.maxspeed*0.75)
                    return 0
                else:
                    print("done adjusting position.")
                    print(f"current odom position: ({self.odomX},{self.odomY})")
                    self.doneParking = True
                    return 0
    
    def exitPark(self):
        if self.doneManeuvering:
            print("done exit maneuvering. Back to lane following...")
            self.doneManeuvering = False #reset
            self.exitDecision = -1 #reset
            self.state = 0 #go back to lane following
            self.initialPoints = None #reset initial points
            self.timerP = None # useless comment
            self.pl = 320
            return 1
        elif self.exitDecision < 0:
            if self.decisionsI >= len(self.decisions):
                self.idle()
                self.idle()
                self.idle()
                rospy.signal_shutdown("Exit")
            self.exitDecision = self.decisions[self.decisionsI] #replace this with service call
            self.decisionsI+=1
            print("exit decision: going ") #+ self.exitDecisions[self.exitDecision])
            if self.exitDecision == 5: #left exit
                self.trajectory = self.left_exit_trajectory
            elif self.exitDecision == 6: #right exit
                self.trajectory = self.right_exit_trajectory
            elif self.exitDecision == 7: #parallel exit
                pass
            else:
                raise ValueError("self.exitDecision id wrong: ",self.exitDecision)
            print("exit decision: going " + self.decisionList[self.exitDecision])
        if self.exitDecision != 7:
            if self.initialPoints is None:
                self.yaw=(self.yaw+3.14159)%(6.28318) #flip Yaw
                self.set_current_angle()
                # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
                # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
                self.initialPoints = np.array([self.x, self.y])
                # print("initialPoints points: ", self.initialPoints)
                self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                self.odomTimer = rospy.Time.now()
                self.intersectionState = 1 if self.intersectionDecision!=2 else 1#adjusting angle:0, trajectory following:1, adjusting angle2: 2..
            self.yaw=(self.yaw+3.14159)%(6.28318) #flip Yaw
            self.odometry()
            poses = np.array([self.odomX,self.odomY])
            poses = poses.dot(self.rotation_matrices[self.orientation])
            x,y = -poses[0], -poses[1]
            # print("position: ",x,y)
            if self.intersectionState==0: #adjusting
                error = self.yaw-self.currentAngle
                if self.yaw>=5.73: #subtract 2pi to get error between -pi and pi
                    error-=6.28
                # print("yaw, curAngle, error: ", self.yaw, self.currentAngle, error)
                if abs(error) <= 0.05:
                    self.intersectionState+=1 #done adjusting
                    print("done adjusting angle. Transitioning to trajectory following")
                    print(f"current position: ({self.odomX},{self.odomY})")
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    return 0
                else:
                    self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
                    return 0
            elif self.intersectionState==1: #trajectory following
                desiredY = self.trajectory(x)
                error = y - desiredY
                # print("x, y_error: ",x,abs(error) )
                arrived = abs(self.yaw-self.destinationAngle) <= 0.05
                # print("yaw_error: ")
                # print(str(self.yaw-self.destinationAngle))
                if arrived:
                    print("trajectory done. adjust angle round 2")
                    self.intersectionState += 1
                    self.last_error2 = 0 #reset pid errors
                    self.error_sum2 = 0
                    return 0
                # steering_angle = self.pid2(error)
                # print("steering: ",steering_angle)
                # print("x, y, desiredY, angle, steer: ", x, y, desiredY, self.yaw, steering_angle*180/3.14159)
                self.publish_cmd_vel(-self.pid2(error), -self.maxspeed*0.9)
                return 0
            elif self.intersectionState == 2: #adjust angle 2
                error = self.yaw-self.destinationAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, destAngle, error: ", self.yaw, self.destinationAngle, error)
                if abs(error) <= 0.05:
                    print("done adjusting angle!!")
                    self.doneManeuvering = True
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    return 0
                else:
                    self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
                    return 0
        elif self.exitDecision == 7:
            if self.initialPoints is None:
                self.set_current_angle()
                # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
                # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
                self.initialPoints = np.array([self.x, self.y])
                # print("initialPoints points: ", self.initialPoints)
                self.offset = 0
                print("begin going straight for "+str(self.offset)+"m")
                self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                self.odomTimer = rospy.Time.now()
                self.intersectionState = 0 #going straight:0, trajectory following:1, adjusting angle2: 2..
            self.odometry()
            poses = np.array([self.odomX, self.odomY])
            poses = poses.dot(self.rotation_matrices[self.orientation])
            x, y = poses[0], poses[1]
            # print("position: ",x,y)
            if self.intersectionState==0: #going straight
                error = self.yaw-self.currentAngle
                if x >= self.offset:
                    print("done going straight. begin adjusting angle...")
                    self.intersectionState = 1
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    # print("current angle, destination: ", self.yaw, self.destinationAngle)
                self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
                return 0
            if self.intersectionState==1: #adjusting
                error = self.yaw - self.destinationAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, error: ", self.yaw, error)
                if abs(error) >= self.parallelParkAngle*np.pi/180:
                    self.intersectionState = 2
                    print(f"{self.parallelParkAngle} degrees...")
                    self.timer5 = rospy.Time.now()+rospy.Duration(3) #change to odom
                self.publish_cmd_vel(-23, self.maxspeed*0.9)
                return 0
            elif self.intersectionState==2: #adjusting
                error = self.yaw - self.destinationAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                if abs(error) < 0.05:
                    print("done adjusting angle!!")
                    self.doneManeuvering = True
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    return 0
                self.publish_cmd_vel(23, self.maxspeed*0.9)
                return 0
    
    def curvedpath(self):
        if self.doneManeuvering:
            print("done curvedpath maneuvering.")
            self.doneManeuvering = False #reset
            self.intersectionDecision = -1 #reset
            self.initialPoints = None #reset initial points
            self.pl = 320
            return 1
        elif self.intersectionDecision < 0:
            if self.decisionsI >= len(self.decisions):
                self.idle()
                self.idle()
                self.idle()
                rospy.signal_shutdown("Exit")
            self.intersectionDecision = self.decisions[self.decisionsI] #replace this with service call
            self.decisionsI+=1
            if self.intersectionDecision == 14:
                pass
            else:
                raise ValueError("self.intersectionDecision id wrong: ",self.intersectionDecision)
            print("highway decision: going " + self.decisionList[self.intersectionDecision])
        if self.initialPoints is None:
            self.set_current_angle()
            # print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
            # print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
            self.initialPoints = np.array([self.x, self.y])
            # print("initialPoints points: ", self.initialPoints)
            self.offset = 0.6 #tune this
            self.odomX, self.odomY = 0.0, 0.0 #reset x,y
            self.odomTimer = rospy.Time.now()
            self.intersectionState = 0
        self.odometry()
        poses = np.array([self.odomX,self.odomY])
        poses = poses.dot(self.rotation_matrices[self.orientation])
        x,y = poses[0], poses[1]
        # print("position: ",x,y)
        if self.intersectionState==0: #adjusting
            error = self.yaw-self.currentAngle
            if self.yaw>=5.73: #subtract 2pi to get error between -pi and pi
                error-=6.28
            # print("yaw, curAngle, error: ", self.yaw, self.currentAngle, error)
            if x >= self.offset:
                print("trajectory done.")
                self.doneManeuvering = True
                self.error_sum = 0 #reset pid errors
                self.last_error = 0
                return 0
            else:
                self.publish_cmd_vel(self.pid(error), self.maxspeed*0.9)
                return 0
    
    #transition events
    def entering_roundabout(self):
        return self.object_detected(3)
    def stop_sign_detected(self):
        return self.object_detected(2)
    def highway_entrance_detected(self):
        return self.object_detected(7)
    def highway_exit_detected(self):
        return self.object_detected(1)
    def light_detected(self):
        return self.object_detected(9)
    def parking_detected(self):
        return self.object_detected(4)
    def is_green(self):
        if not self.simulation: #if not simulation consider red
            return False
        else:
            self.orientation = np.argmin([abs(self.yaw),abs(self.yaw-1.5708),abs((self.yaw)-3.14159),abs(self.yaw-4.71239),abs(self.yaw-6.28319)])%4
            if self.orientation==1 or self.orientation==3: #N or S
                topic = 'start'
            else:
                topic = 'master'
            state=rospy.wait_for_message('/automobile/trafficlight/'+topic,Byte)#0=red,1=yellow,2=green
            return True if state.data == 2 else False
    def crosswalk_sign_detected(self):
        return self.object_detected(5)
    def pedestrian_appears(self):
        return self.object_detected(11)
    def car_detected(self):
        return self.object_detected(12)

    #controller functions
    def idle(self):
        # self.cmd_vel_pub(0.0, 0.0)
        # self.msg.data = '{"action":"3","brake (steerAngle)":'+str(0.0)+'}'
        # self.cmd_vel_pub.publish(self.msg)
        self.msg.data = '{"action":"1","speed":'+str(0.0)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(0.0)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)

    #odom helper functions
    def pid(self, error):
        # self.error_sum += error * self.dt
        dt = (rospy.Time.now()-self.timer4).to_sec()
        # rospy.loginfo("time: %.4f", self.dt)
        self.timer4 = rospy.Time.now()
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.kd * derivative #+ self.ki * self.error_sum
        self.last_error = error
        return output
    def pid2(self, error):
        # self.error_sum2 += error * self.dt
        dt = (rospy.Time.now()-self.timer5).to_sec()
        # rospy.loginfo("time: %.4f", self.dt)
        self.timer4 = rospy.Time.now()
        derivative = (error - self.last_error2) / dt
        output = self.kp2 * error + self.kd2 * derivative #+ self.ki2 * self.error_sum2
        self.last_error2 = error
        return output
    def odometry(self):
        dt = (rospy.Time.now()-self.odomTimer).to_sec()
        self.odomTimer = rospy.Time.now()
        magnitude = self.velocity*dt*self.odomRatio
        self.odomX += magnitude * math.cos(self.yaw)
        self.odomY += magnitude * math.sin(self.yaw)
        # print(f"odometry: speed={self.velocity}, dt={dt}, mag={magnitude}, cos={math.cos(self.yaw)}, X={self.odomX}, Y={self.odomY}")
    def set_current_angle(self):
        #0:left, 1:straight, 2:right, 3:parkF, 4:parkP, 5:exitparkL, 6:exitparkR, 7:exitparkP
        #8:enterhwLeft, 9:enterhwStright, 10:rdb, 11:exitrdbE, 12:exitrdbS, 13:exitrdbW, 14:curvedpath
        self.orientation = np.argmin([abs(self.yaw),abs(self.yaw-1.5708),abs((self.yaw)-3.14159),abs(self.yaw-4.71239),abs(self.yaw-6.28319)])%4
        self.currentAngle = self.orientations[self.orientation]
        if self.intersectionDecision == 0 or self.exitDecision == 5: #left
            self.destinationOrientation = self.directions[(self.orientation+1)%4]
            self.destinationAngle = self.orientations[(self.orientation+1)%4]
            return
        elif self.intersectionDecision == 1: #straight
            self.destinationOrientation = self.orientation
            self.destinationAngle = self.currentAngle
            return
        elif self.intersectionDecision == 2 or self.parkingDecision == 3 or self.exitDecision == 6: #right
            self.destinationOrientation = self.directions[(self.orientation-1)%4]
            self.destinationAngle = self.orientations[(self.orientation-1)%4]
            return
        elif self.parkingDecision == 4:
            self.destinationOrientation = self.directions[(self.orientation)%4]
            self.destinationAngle = self.orientations[(self.orientation)%4]
            return

    #trajectories
    def straight_trajectory(self, x):
        return 0
    def left_trajectory_real(self, x):
        return math.exp(3.57*x-4.3)
    def right_trajectory_real(self, x):
        return -math.exp(4*x-3.05)
    def left_exit_trajectory_real(self, x):
        return math.exp(4*x-3.05)
    def right_exit_trajectory_real(self, x):
        return -math.exp(4*x-3.05)
    def leftpark_trajectory(self, x):
        return math.exp(3.57*x-4.2) #real dimensions
    def left_trajectory_sim(self, x):
        return math.exp(3.57*x-4.33)
    def right_trajectory_sim(self, x):
        return -math.exp(3.75*x-3.33)
    def left_exit_trajectory_sim(self, x):
        return math.exp(3.75*x-3.03)
    def right_exit_trajectory_sim(self, x):
        return -math.exp(3.75*x-3.03)
    def rdb_trajectory(self, x, t):
        u = 0.5-math.pow(x-0.71,2)
        u = np.clip(u,0,255)
        yaw = self.yaw+t if self.yaw+t>0 else (6.2831853+self.yaw+t)
        return -math.sqrt(u)+0.25 if yaw<np.pi/2 or yaw>3*np.pi/2 else math.sqrt(u)+0.25

    #others
    def object_detected(self, obj_id):
        if self.numObj >= 2:
            if self.detected_objects[0]==obj_id: 
                if self.check_size(obj_id,0):
                    return True
            elif self.detected_objects[1]==obj_id:
                if self.check_size(obj_id,1):
                    return True
        elif self.numObj == 1:
            if self.detected_objects[0]==obj_id: 
                if self.check_size(obj_id,0):
                    return True
        return False
    def check_size(self, obj_id, index):
        #checks whether a detected object is within a certain min and max sizes defined by the obj type
        box = self.box1 if index==0 else self.box2
        size = max(box[2], box[3])
        if obj_id==12:
            size = min(box[2], box[3])
        return size >= self.min_sizes[obj_id] and size <= self.max_sizes[obj_id]
    def get_steering_angle(self):
        """
        Determine the steering angle based on the lane center
        :param center: lane center
        :return: Steering angle in radians
        """
        # Calculate the steering angle in radians
        image_center = 640 / 2 
        error = (self.center - image_center)
        d_error = (error-self.last)/self.dt
        self.last = error
        steering_angle = (error*self.p+d_error*self.d)
        return steering_angle
    def publish_cmd_vel(self, steering_angle, velocity = None, clip = True):
        """
        Publish the steering command to the cmd_vel topic
        :param steering_angle: Steering angle in radians
        """
        if velocity is None:
            velocity = self.maxspeed
        if clip:
            steering_angle = np.clip(steering_angle*180/np.pi, -22.9, 22.9)
        self.msg.data = '{"action":"1","speed":'+str(velocity)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(float(steering_angle))+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)

    #PID tuning
    def trackbars(self):
        windowName = "Params"
        image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/map.png')
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,480,360)
        cv2.createTrackbar('Save',windowName,0,1,self.save_object)
        cv2.createTrackbar('View',windowName,0,1,self.view)
        cv2.createTrackbar('p',windowName,int(self.p*100000),600,self.changep)
        cv2.createTrackbar('d',windowName,int(self.d*100000),50,self.changed)
        cv2.createTrackbar('i',windowName,int(self.i*100000),100,self.changei)
        # cv2.createTrackbar('kp',windowName,int(self.kp*1000),2000,self.changekp)
        # cv2.createTrackbar('kd',windowName,int(self.kd*1000),1000,self.changekd)
        # cv2.createTrackbar('ki',windowName,int(self.ki*1000),1000,self.changeki)
        # cv2.createTrackbar('kp2',windowName,int(self.kp2*1000),2000,self.changekp2)
        # cv2.createTrackbar('kd2',windowName,int(self.kd2*1000),2000,self.changekd2)
        # cv2.createTrackbar('ki2',windowName,int(self.ki2*1000),2000,self.changeki2)
        cv2.imshow(windowName, image)
        key = cv2.waitKey(0)
    def save_object(self,v):
        file = open(os.path.dirname(os.path.realpath(__file__))+'/PID.json', 'w')
        data = {"p":self.p,"d":self.d,"i":self.i,"kp":self.kp,"kd":self.kd,"ki":self.ki,"kp2":self.kp2,"kd2":self.kd2,"ki2":self.ki2}
        json.dump(data, file)
        self.view(0)
    def view(self,v):
        print("=========== PIDS ============"+'\n'+
            "p           "+str(self.p)+
            "\nd         "+str(self.d)+
            "\ni         "+str(self.i)+
            "\nkp         "+str(self.kp)+
            "\nkd         "+str(self.kd)+
            "\nki         "+str(self.ki)+
            "\nkp2        "+str(self.kp2)+
            "\nkd2        "+str(self.kd2)+
            "\nki2        "+str(self.ki2)        
        )
    def changep(self,v):
        self.p = v/100000
    def changed(self,v):
        self.d = v/100000
    def changei(self,v):
        self.i = v/100000
    def changekp(self,v):
        self.kp = v/1000
    def changekd(self,v):
        self.kd = v/1000
    def changeki(self,v):
        self.ki = v/1000
    def changekp2(self,v):
        self.kp2 = v/1000
    def changekd2(self,v):
        self.kd2 = v/1000
    def changeki2(self,v):
        self.ki2 = v/1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='State Machine for Robot Control.')
    parser.add_argument("--simulation", type=str, default=True, help="Run the robot in simulation or real life")
    parser.add_argument("--path", type=str, default="/paths/path.json", help="Planned path")
    parser.add_argument("--custom", type=str, default=False, help="Custom path")
    # args, unknown = parser.parse_known_args()
    args = parser.parse_args(rospy.myargv()[1:])
    if args.simulation=="True":
        s = True
    else:
        s = False
    if args.custom=="True":
        c = True
    else:
        c = False
    node = StateMachine(simulation=s,planned_path=args.path,custom_path=c)
    # rospy.spin()
    node.callback_thread.join()
    node.action_thread.join()