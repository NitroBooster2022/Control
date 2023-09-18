#!/usr/bin/env python3
#added odometry
import rospy
import numpy as np
from message_filters import ApproximateTimeSynchronizer
from std_msgs.msg import String, Float32
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Vector3
from utils.msg import Lane, Sign, localisation, IMU
from pynput import keyboard
from utils.srv import get_direction, nav
import message_filters
import time
import math

import cv2
import os
import json


class StateMachine():
    def __init__(self):
        #states
        self.states = ['Lane Following', "Approaching Intersection", "Stopping at Intersection", 
                       "Intersection Maneuvering", "Approaching Crosswalk", "Pedestrian", "Highway",
                       "Carblock", "Roundabout", "Parking", "initial"]
        self.state = 10

        #sign
        self.class_names = ['oneway', 'highwayexit', 'stopsign', 'roundabout', 'park', 'crosswalk', 'noentry', 'highwayentrance', 'priority',
                'lights','block','pedestrian','car','others','nothing']
        self.min_sizes = [00,00,20,00,00,52,00,00,00,150,00,000,90]
        self.max_sizes = [50,50,30,50,50,65,50,50,60,188,50,100,125]
        self.center = -1
        self.detected_objects = []
        self.numObj = -1
        self.box1 = []
        self.box2 = []

        #pose
        self.x = 0.82
        self.y = 14.91
        self.poses = np.array([self.x,self.y])
        self.yaw = 1.5707
        self.speed = 0
        self.odomX = 0
        self.odomY = 0

        #PID
        #for initial angle adjustment
        self.kp = 1.57
        self.ki = 0.005
        self.kd = 0.3
        self.error_sum = 0
        self.last_error = 0
        #for goal points
        self.kp2 = 1.57
        self.ki2 = 0.00#5
        self.kd2 = 0.0#8
        self.error_sum2 = 0
        self.last_error2 = 0

        #steering
        self.msg = String()
        self.msg2 = String()
        self.p = 0.005
        self.ArrivedAtStopline = False
        self.maxspeed = 0.15
        self.i = 0
        self.d = 0.000#15
        self.last = 0
        self.center = 0

        #timers
        self.timer = None
        self.timer2 = None
        self.timer3 = None
       
        self.toggle = 0 #True: velocity; False: steering angle
        print("hello world")

        #intersection
        self.intersectionStop = None
        self.intersectionDecision = -1 #0:left, 1:straight, 2:right
        self.intersectionDecisions = ["left", "straight", "right"] #0:left, 1:straight, 2:right
        self.doneManeuvering = False
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
        self.right_offset_x = 0.85
        self.right_offset_y = 0.573
        self.offsets_x = np.array([self.left_offset_x, self.left_offset_x*1.2, self.right_offset_x])
        self.rotation_matrices = np.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]]]) #E,N,W,S

        #carblock
        self.carCleared = None
        self.dotted = False
        """
        Initialize the lane follower node
        """
        rospy.init_node('lane_follower_node', anonymous=True)
        self.timer4 = rospy.Time.now()
        self.timer5 = rospy.Time.now()
        self.timer6 = rospy.Time.now()
        self.odomTimer = rospy.Time.now()
        self.cmd_vel_pub = rospy.Publisher("/automobile/command", String, queue_size=3)
        self.rate = rospy.Rate(50)
        self.dt = 1/50 #for PID

        # Create service proxy
        self.get_dir = rospy.ServiceProxy('get_direction',get_direction)

        # Subscribe to topics
        self.lane_sub = message_filters.Subscriber('lane', Lane, queue_size=3)
        self.sign_sub = message_filters.Subscriber('sign', Sign, queue_size=3)
        # self.localization_sub = message_filters.Subscriber("/automobile/localisation", localisation, queue_size=3)
        self.imu_sub = message_filters.Subscriber("/automobile/IMU", IMU, queue_size=3)
        # self.encoder_sub = message_filters.Subscriber("/automobile/encoder", Float32, queue_size=3)
        # self.encoder_sub = message_filters.Subscriber("/gazebo/model_states", ModelStates, queue_size=3)
        self.subscribers = []
        self.subscribers.append(self.lane_sub)
        self.subscribers.append(self.sign_sub)
        # self.subscribers.append(self.localization_sub)
        self.subscribers.append(self.imu_sub)

        # self.subscribers.append(self.encoder_sub)
        # self.subscribers.append(self.encoder_sub)

        
        # Create an instance of TimeSynchronizer
        ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=3, slop=1000)
        ts.registerCallback(self.callback)

        # self.trackbars()
        
        # Load data
        file = open(os.path.dirname(os.path.realpath(__file__))+'/PID.json', 'r')
        data = json.load(file)
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

    def trackbars(self):
        windowName = "Params"
        image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/map.png')
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,480,360)
        cv2.createTrackbar('Save',windowName,0,1,self.save_object)
        cv2.createTrackbar('View',windowName,0,1,self.view)
        cv2.createTrackbar('p',windowName,int(self.p*1000),30,self.changep)
        cv2.createTrackbar('d',windowName,int(self.d*1000),100,self.changed)
        cv2.createTrackbar('i',windowName,int(self.i*1000),100,self.changei)
        cv2.createTrackbar('kp',windowName,int(self.kp*1000),2000,self.changekp)
        cv2.createTrackbar('kd',windowName,int(self.kd*1000),1000,self.changekd)
        cv2.createTrackbar('ki',windowName,int(self.ki*1000),1000,self.changeki)
        cv2.createTrackbar('kp2',windowName,int(self.kp2*1000),2000,self.changekp2)
        cv2.createTrackbar('kd2',windowName,int(self.kd2*1000),2000,self.changekd2)
        cv2.createTrackbar('ki2',windowName,int(self.ki2*1000),2000,self.changeki2)
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
        self.p = v/1000
    def changed(self,v):
        self.d = v/1000
    def changei(self,v):
        self.i = v/1000
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
    
    #callback function
    def callback(self,lane,sign,imu):
        self.dt = (rospy.Time.now()-self.timer6).to_sec()
        # rospy.loginfo("time: %.4f", self.dt)
        self.timer6 = rospy.Time.now()
        # Perform decision making tasks
        # Compute the steering angle & linear velocity
        # Publish the steering angle & linear velocity to the /automobile/command topic
        # self.x = localization.posA
        # self.y = 15.0-localization.posB

        self.yaw = imu.yaw if imu.yaw>0 else (6.2831853+imu.yaw)

        # self.poses[0] = self.x
        # self.poses[1] = self.y
        # x_speed = encoder.twist[72].linear.x
        # y_speed = encoder.twist[72].linear.y
        # print(x_speed, y_speed)
        
        # print("x,y,yaw: ", self.x, self.y, self.yaw)
        self.center = lane.center
        self.detected_objects = sign.objects
        self.numObj = sign.num
        self.box1 = sign.box1
        self.box2 = sign.box2
        self.ArrivedAtStopline = lane.stopline
        # for i in range(self.numObj):
        #     if i == 0:
        #         print(self.numObj)
        #         print(f"{self.class_names[self.detected_objects[i]]} detected! width, height: {self.box1[2]}, {self.box1[3]}")
        #     elif i == 1:
        #         print(f"{self.class_names[self.detected_objects[i]]} detected! width, height: {self.box2[2]}, {self.box2[3]}")
        #     else:
        #         print(f"{self.class_names[self.detected_objects[i]]} detected!")
        act = self.action()
        if int(act)==1:
            print(f"transitioning to '{self.states[self.state]}'")
        # print("time: ",time.time()-t1)

    #state machine
    def action(self):
        if self.state==0: #lane following
            # Determine the steering angle based on the center
            steering_angle = self.get_steering_angle(self.center)
            # Publish the steering command
            self.publish_cmd_vel(steering_angle) 
            #transition events
            if self.ArrivedAtStopline:
                print("signless intersection detected... -> state 3")
                self.doneManeuvering = False #set to false before entering state 3
                self.state = 3
                return 1
            if self.stop_sign_detected():
                print("stop sign detected -> state 1")
                self.intersectionStop = True
                self.state = 1
                return 1
            elif self.light_detected():
                #call service to check light color
                if self.is_green():
                    print("green light detected -> state 1")
                    self.intersectionStop = False
                else:
                    print("red light detected -> state 1")
                    self.intersectionStop = True
                self.state = 1
                return 1
            elif self.crosswalk_sign_detected():
                print("crosswalk sign detected -> state 4")
                self.state = 4
                return 1
            elif self.pedestrian_appears():
                print("pedestrian appears!!! -> state 5")
                self.state = 5
                return 1
            elif self.highway_entrance_detected():
                print("entering highway -> 6")
                self.state = 6
                return 1
            elif self.entering_roundabout():
                print("entering roundabout -> 8")
                self.state = 8
                return 1
            elif self.can_park():
                print("about to park -> 9")
                self.state = 9
                return 1
            return 0
        elif self.state == 1: #Approaching Intersection
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
            #Action: Adjust Position
            # Determine the steering angle based on the center
            steering_angle = self.get_steering_angle(self.center)
            # Publish the steering command
            self.publish_cmd_vel(steering_angle) 
            return 0
        elif self.state == 2: #Stopping at Intersection
            #Action: idle
            self.idle()
            #Transition events
            if self.timer is None:
                self.timer = rospy.Time.now() + rospy.Duration(3.57)
            elif rospy.Time.now() >= self.timer:
                self.timer = None
                self.doneManeuvering = False #set to false before entering state 3
                self.state = 3
                return 1
            return 0
        elif self.state == 3: #Intersection Maneuvering
            # print("state 3")
            #go left, straight or right
            #Transition Events
            if self.doneManeuvering:
                print("done intersection maneuvering. Back to lane following...")
                self.doneManeuvering = False #reset
                self.intersectionDecision = -1 #reset
                self.state = 0 #go back to lane following
                self.initialPoints = None #reset initial points
                return 1
            elif self.intersectionDecision <0: 
                # self.intersectionDecision = np.random.randint(low=0, high=3) #replace this with service call
                self.intersectionDecision = 0 #always left
                print("intersection decision: going " + self.intersectionDecisions[self.intersectionDecision])
                if self.intersectionDecision == 0: #left
                    self.trajectory = self.left_trajectory
                elif self.intersectionDecision == 1: #straight
                    self.trajectory = self.straight_trajectory
                else:
                    self.trajectory = self.right_trajectory
            if self.initialPoints is None:
                self.set_current_angle()
                print("current orientation: ", self.directions[self.orientation], self.orientations[self.orientation])
                print("destination orientation: ", self.destinationOrientation, self.destinationAngle)
                self.initialPoints = np.array([self.x, self.y])
                print("initialPoints points: ", self.initialPoints)
                print("begin adjusting angle...")
                self.odomX, self.odomY = 0.0, 0.0 #reset x,y
                self.odomTimer = rospy.Time.now()
                self.intersectionState = 0 #adjusting angle:0, trajectory following:1, adjusting angle2: 2..
            self.odometry()
            if self.intersectionState==0: #adjusting
                error = self.yaw-self.currentAngle 
                if self.yaw>=5.73: #subtract 2pi to get error between -pi and pi
                    error-=6.28
                # print("yaw, curAngle, error: ", self.yaw, self.currentAngle, error)
                if error <= 0.05:
                    self.intersectionState+=1 #done adjusting
                    print("done adjusting angle. Transitioning to trajectory following")
                    print(f"current position: ({self.odomX},{self.odomY})")
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    return 0
                else:
                    steering_angle = self.pid(error)
                    self.publish_cmd_vel(steering_angle, self.maxspeed*0.3)
                    return 0
            elif self.intersectionState==1: #trajectory following
                poses = np.array([self.odomX,self.odomY])
                # print("subtracted by offsets: ", poses)
                poses = poses.dot(self.rotation_matrices[self.orientation])
                # print("after transformation: ", poses)
                x,y = poses[0], poses[1]
                desiredY = self.trajectory(x)
                error = y - desiredY
                if x>=(self.offsets_x[self.intersectionDecision]-0.1) and (abs(error)<=0.0753):
                    print("trajectory done. adjust angle round 2")
                    self.intersectionState += 1
                    self.last_error2 = 0 #reset pid errors
                    self.error_sum2 = 0
                    return 0
                steering_angle = self.pid2(error)
                # print("x, y, desiredY, angle: ", x, y, desiredY, steering_angle*57.29578)
                self.publish_cmd_vel(steering_angle, self.maxspeed*0.75)
                return 0
            elif self.intersectionState == 2: #adjust angle 2
                error = self.yaw-self.destinationAngle
                if self.yaw>=5.73: #subtract 2pi to get small error
                    error-=6.28
                # print("yaw, destAngle, error: ", self.yaw, self.destinationAngle, error)
                if abs(error) <= 0.25:
                    print("done adjusting angle!!")
                    self.doneManeuvering = True
                    self.error_sum = 0 #reset pid errors
                    self.last_error = 0
                    return 0
                else:
                    steering_angle = self.pid(error)
                    self.publish_cmd_vel(steering_angle, self.maxspeed*0.5)
                    return 0
        elif self.state == 4: #Approaching Crosswalk
            #Transition events
            if self.timer is None: #start timer. ~10 seconds to pass crosswalk?
                self.timer = rospy.Time.now() + rospy.Duration(10)
            if rospy.Time.now() >= self.timer:
                self.timer = None #reset timer
                self.state = 0
                return 1
            elif self.pedestrian_appears():
                self.timer = None #reset timer
                self.state = 5
                return 1
            #Action: slow down
            steering_angle = self.get_steering_angle(self.center)
            # Publish the steering command
            self.publish_cmd_vel(steering_angle, self.maxspeed*0.75)
            return 0
        elif self.state == 5: #Pedestrian
            if self.pedestrian_clears():
                self.state = 0
                return 1
            #Action: idle
            self.idle()
            return 0
        elif self.state == 6: #Highway
            if self.highway_exit_detected():
                if self.entering_roundabout():
                    self.state = 8
                else:
                    self.state = 0
            steering_angle = self.get_steering_angle(self.center)
            self.publish_cmd_vel(steering_angle, self.maxspeed*1.33) 
        elif self.state == 7: #Carblock
            #/entry: checkDotted
            #action: overtake or wait
            if self.carCleared is None:
                self.carCleared = False
            if self.dotted:
                #overtake
                if self.timer is None and self.timer2 is None and self.timer3 is None: #begin going left
                    print("begin going left")
                    self.timer = rospy.Time.now()+rospy.Duration(2.0)
                if self.timer is not None and self.timer2 is None and self.timer3 is None:
                    if rospy.Time.now() >= self.timer: #finished going straight. reset timer to None
                        print("finished going left. reset timer to None")
                        self.timer = None
                        self.timer2 = rospy.Time.now()+rospy.Duration(3.0)
                    else:
                        self.left()
                        return 0
                if self.timer is None and self.timer2 is not None and self.timer3 is None: #begin going straight
                    if rospy.Time.now() >= self.timer2: #finished going straight
                        print("finished going straight. reset timer2 to None. back to lane")
                        self.timer2 = None #finished going left. reset timer2 to None.
                        self.timer3 = rospy.Time.now()+rospy.Duration(2.0)
                        return 0
                    else: 
                        self.straight()
                        return 0 
                if self.timer is None and self.timer2 is None and self.timer3 is not None: #go back to lane
                    if rospy.Time.now() >= self.timer3: #finished going straight
                        print("done overtaking. back to lane following")
                        self.timer3 = None #finished going left. reset timer2 to None.
                        self.carCleared = True
                        return 0
                    else: 
                        self.left()
                        return 0 
            else: #wait
                if self.object_detected(12):
                    self.idle()
                else:
                    self.carCleared = True
                return 0
        elif self.state == 8: #Roundabout
            self.state = 0
        elif self.state == 9: #Parking
            self.state = 0
        elif self.state == 10: #initialization state
            if self.timer is None:
                print("initializing...")
                self.toggle = 0
                self.timer = rospy.Time.now() + rospy.Duration(1.57)
            if rospy.Time.now() >= self.timer:
                print("done initializing.")
                self.timer = None
                self.state = 0
                self.toggle = True
                return 1
            else:
                if self.toggle == 0:
                    self.toggle = 1
                    self.msg.data = '{"action":"4","activate": true}'
                elif self.toggle == 1: 
                    self.toggle = 2
                    self.msg.data = '{"action":"1","speed":'+str(0.0)+'}'
                elif self.toggle == 2:
                    self.toggle = 0
                    self.msg.data = '{"action":"5","activate": true}'
                self.cmd_vel_pub.publish(self.msg)
        return 0
    #Transition events
    def can_park(self):
        return False
    def stopline_detected(self):
        return False
    def entering_roundabout(self):
        return False
    def stop_sign_detected(self):
        return self.object_detected(2)
    def highway_entrance_detected(self):
        return self.object_detected(7)
    def highway_exit_detected(self):
        return self.object_detected(1)
    def light_detected(self):
        return self.object_detected(9)
    def done_stopping(self):
        #set a timer of 3.57 seconds 
        if self.timer is None:
            self.timer = rospy.Time.now() + rospy.Duration(3.57)
        elif rospy.Time.now() >= self.timer:
            self.timer = None
            return True
        return False
    def arrived_at_intersection(self):
        return self.ArrivedAtStopline
    def is_green(self): #call service or message
        return False
        # r=self.get_dir(0,0,0,'').dir
        # if r.split()[2]=='N' or r.split()[2]=='S':
        #     topic = 'start'
        # else:
        #     topic = 'master'
        # state=rospy.wait_for_message('/automobile/trafficlight/'+topic,Byte)
        # if state.data==0:
        #     print('redlight')
        #     return False
        # elif state.data==1:
        #     print('yellowlight')
        #     return False
        # else:
        #     return True
    def crosswalk_sign_detected(self):
        return self.object_detected(5)
    def pedestrian_appears(self):
        return False
    def passed_crosswalk(self):
        return False
    def pedestrian_clears(self):
        return False

    #controller functions
    def straight(self):
        # self.cmd_vel_pub(0.0, 0.2)
        self.msg.data = '{"action":"1","speed":'+str(0.2)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(0.0*180/np.pi)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
    def left(self):
        # self.cmd_vel_pub(-23, 0.12)
        self.msg.data = '{"action":"1","speed":'+str(0.12)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(-23*180/np.pi)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
    def right(self):
        # self.cmd_vel_pub(23, 0.12)
        self.msg.data = '{"action":"1","speed":'+str(0.12)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(23*180/np.pi)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
    def idle(self):
        # self.cmd_vel_pub(0.0, 0.0)
        self.msg.data = '{"action":"1","speed":'+str(0.0)+'}'
        # self.msg2.data = '{"action":"2","steerAngle":'+str(0.0)+'}'
        self.cmd_vel_pub.publish(self.msg)
        # self.cmd_vel_pub.publish(self.msg2)
    def go_back(self):
        # self.cmd_vel_pub(0.0, -0.2)
        self.msg.data = '{"action":"1","speed":'+str(-0.2)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(0.0)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)

    #helper functions
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
        magnitude = self.maxspeed*dt*0.75
        # print(f"magnitude*cos: {magnitude * math.cos(self.yaw)}, odomX: {self.odomX}")
        self.odomX += magnitude * math.cos(self.yaw)
        self.odomY += magnitude * math.sin(self.yaw)
        # print(f"odometry: speed={self.maxspeed*0.75}, dt={dt}, mag={magnitude}, cos={math.cos(self.yaw)}, X={self.odomX}, Y={self.odomY}")
    def left_trajectory(self, x):
        return math.exp(3.57*x-4.3)
    def straight_trajectory(self, x):
        return 0
    def right_trajectory(self, x):
        return -math.exp(3.57*x-3.57)
    def set_current_angle(self):
        self.orientation = np.argmin([abs(self.yaw),abs(self.yaw-1.5708),abs((self.yaw)-3.14159),abs(self.yaw-4.71239),abs(self.yaw-6.28319)])%4
        self.currentAngle = self.orientations[self.orientation]
        if self.intersectionDecision == 0: #left
            self.destinationOrientation = self.directions[(self.orientation+1)%4]
            self.destinationAngle = self.orientations[(self.orientation+1)%4]
            return
        elif self.intersectionDecision == 1: #straight
            self.destinationOrientation = self.orientation
            self.destinationAngle = self.currentAngle
            return
        elif self.intersectionDecision == 2: #right
            self.destinationOrientation = self.directions[(self.orientation-1)%4]
            self.destinationAngle = self.orientations[(self.orientation-1)%4]
            return
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
        return size >= self.min_sizes[obj_id] and size <= self.max_sizes[obj_id]
    def get_steering_angle(self, center):
        """
        Determine the steering angle based on the lane center
        :param center: lane center
        :return: Steering angle in radians
        """
        # Calculate the steering angle in radians
        image_center = 640 / 2

        if center==image_center:
            center=self.center
            self.center=image_center
        else:
            self.center=center
        
        error = (center - image_center)
        d_error = (error-self.last)/self.dt
        self.last = error
        steering_angle = (error * self.p+d_error*self.d)

        return steering_angle

    def publish_cmd_vel(self, steering_angle, velocity = None, clip = False):
        """
        Publish the steering command to the cmd_vel topic
        :param steering_angle: Steering angle in radians
        """
        if clip:
            steering_angle = np.clip(steering_angle, -0.4, 0.4)
        self.msg.data = '{"action":"1","speed":'+str(self.maxspeed)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(steering_angle*180/np.pi)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
        # if velocity is None:
        #     velocity = self.maxspeed + self.maxspeed*abs(steering_angle)/0.4
        # if self.toggle:
        #     self.toggle = False
        #     self.msg.data = '{"action":"1","speed":'+str(velocity)+'}'
        # else:
        #     self.toggle = True
        #     if clip:
        #         steering_angle = np.clip(steering_angle, -0.4, 0.4)
        #     self.msg.data = '{"action":"2","steerAngle":'+str(steering_angle*180/np.pi)+'}'
        # self.cmd_vel_pub.publish(self.msg)

if __name__ == '__main__':
    node = StateMachine()
    while not rospy.is_shutdown():
        node.rate.sleep()
        rospy.spin()
