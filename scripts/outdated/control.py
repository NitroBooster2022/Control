#!/usr/bin/env python3
import rospy
import numpy as np
from message_filters import ApproximateTimeSynchronizer
from std_msgs.msg import String, Byte
from utils.msg import Lane, Sign
from pynput import keyboard
from utils.srv import get_direction, nav
import message_filters
import time

class LaneFollower():
    def __init__(self):
        #states
        self.states = ['Lane Following', "Approaching Intersection", "Stopping at Intersection", "Intersection Maneuvering", "Approaching Crosswalk", "Pedestrian", "initial"]
        self.state = 6

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

        #steering
        self.msg = String()
        self.msg2 = String()
        self.p = 0.005
        self.ArrivedAtStopline = False
        self.maxspeed = 0.1
        self.i = 0
        self.d = 0.001
        self.last = 0
        self.center = 0

        #other
        self.timer = None
        self.timer2 = None
        self.intersectionStop = None
        self.intersectionDecision = -1 #0:left, 1:straight, 2:right
        self.intersectionDecisions = ["left", "straight", "right"] #0:left, 1:straight, 2:right
        self.doneManeuvering = False
        """
        Initialize the lane follower node
        """
        rospy.init_node('lane_follower_node', anonymous=True)
        self.cd = rospy.Time.now()
        self.cmd_vel_pub = rospy.Publisher("/automobile/command", String, queue_size=3)
        self.rate = rospy.Rate(20)

        # Create service proxy
        self.get_dir = rospy.ServiceProxy('get_direction',get_direction)

        # Subscribe to topics
        self.lane_sub = message_filters.Subscriber('lane', Lane, queue_size=3)
        self.sign_sub = message_filters.Subscriber('sign', Sign, queue_size=3)
        
        # Create an instance of TimeSynchronizer
        ts = ApproximateTimeSynchronizer([self.lane_sub, self.sign_sub], queue_size=3, slop=0.15)
        ts.registerCallback(self.callback)
    
    #callback function
    def callback(self,lane,sign):
        # Perform decision making tasks
        # Compute the steering angle & linear velocity
        # Publish the steering angle & linear velocity to the /automobile/command topic
        t1 = time.time()
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

    #state machine
    def action(self):
        
        # activate lane follow only
        if self.state!=6:
            self.state = 0

        if self.state==0: #lane following
            # Determine the steering angle based on the center
            steering_angle = self.get_steering_angle(self.center)
            # Publish the steering command
            self.publish_cmd_vel(steering_angle) 
            #transition events
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
            if self.ArrivedAtStopline:
                print("signless intersection detected... -> state 3")
                self.doneManeuvering = False #set to false before entering state 3
                self.state = 3
            return 0
        elif self.state == 1: #Approaching Intersection
            #Transition events
            if self.ArrivedAtStopline:
                if self.intersectionStop:
                    self.state = 2
                    return 1
                else:
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
                print("initializing timer for intersection stop...")
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
                self.doneManeuvering = False #reset
                self.intersectionDecision = -1 #reset
                self.state = 0 #go back to lane following
                return 1
            elif self.intersectionDecision <0: 
                self.intersectionDecision = 0 #replace this with service call
                # self.intersectionDecision = np.random.randint(low=0, high=3) #replace this with service call
                print("intersection decision: going " + self.intersectionDecisions[self.intersectionDecision])
            if self.intersectionDecision == 0: #left
                #go straight for 3.5s then left for 4s
                if self.timer is None and self.timer2 is None: #begin going straight
                    print("begin going straight")
                    self.timer = rospy.Time.now()+rospy.Duration(2.5)
                if self.timer is not None and self.timer2 is None:
                    if rospy.Time.now() >= self.timer: #finished going straight. reset timer to None
                        print("finished going straight. reset timer to None")
                        self.timer = None
                        self.timer2 = rospy.Time.now()+rospy.Duration(4.5)
                    else:
                        self.straight()
                        return 0
                if self.timer is None and self.timer2 is not None: #begin going left
                    if rospy.Time.now() >= self.timer2: #finished going left
                        print("finished going left. reset timer2 to None. Maneuvering done")
                        self.timer2 = None #finished going left. reset timer2 to None.
                        self.doneManeuvering = True
                        return 0
                    else: 
                        self.left()
                        return 0 
            elif self.intersectionDecision == 1: #straight
                #go straight for 3s
                if self.timer is None: #begin going straight
                    print("begin going straight")
                    self.timer = rospy.Time.now()+rospy.Duration(3.7)
                if rospy.Time.now() >= self.timer: #finished going straight. reset timer to None
                    print("finished going straight. reset timer to None")
                    self.timer = None
                    self.doneManeuvering = True
                    return 0
                else:
                    self.straight()
                    return 0
            elif self.intersectionDecision == 2: #right
                #go straight for 1.5s then right for 8s
                if self.timer is None and self.timer2 is None: #begin going straight
                    print("begin going straight")
                    self.timer = rospy.Time.now()+rospy.Duration(3.2)
                if self.timer is not None and self.timer2 is None:
                    if rospy.Time.now() >= self.timer: #finished going straight. reset timer to None
                        print("finished going straight. reset timer to None")
                        self.timer = None
                        self.timer2 = rospy.Time.now()+rospy.Duration(2.3)
                    else:
                        self.straight()
                        return 0
                if self.timer is None and self.timer2 is not None: #begin going left
                    if rospy.Time.now() >= self.timer2: #finished going straight
                        print("finished going left. reset timer2 to None. Maneuvering done")
                        self.timer2 = None #finished going left. reset timer2 to None.
                        self.doneManeuvering = True
                        return 0
                    else: 
                        self.right()
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
            self.publish_cmd_vel(steering_angle, self.maxspeed/2)
            return 0
        elif self.state == 5: #Pedestrian
            if self.pedestrian_clears():
                self.state = 0
                return 1
            #Action: idle
            self.idle()
            return 0
        elif self.state == 6:
            if self.timer is None:
                print("initializing...")
                self.timer = rospy.Time.now() + rospy.Duration(1.57)
                self.toggle = 0
            if rospy.Time.now() >= self.timer:
                print("done initializing.")
                self.timer = None
                self.state = 0
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
        return 0
    #Transition events
    def stopline_detected(self):
        return False
    def stop_sign_detected(self):
        return self.object_detected(2)
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
        self.msg2.data = '{"action":"2","steerAngle":'+str(0.0)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
    def left(self):
        # self.cmd_vel_pub(-23, 0.12)
        self.msg.data = '{"action":"1","speed":'+str(0.12)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(-23.0)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
    def right(self):
        # self.cmd_vel_pub(23, 0.12)
        self.msg.data = '{"action":"1","speed":'+str(0.12)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(23.0)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
    def idle(self):
        # self.cmd_vel_pub(0.0, 0.0)
        self.msg2.data = '{"action":"3","brake (steerAngle)":'+str(0.0)+'}'
        self.cmd_vel_pub.publish(self.msg2)
        # self.msg.data = '{"action":"1","speed":'+str(0.0)+'}'
        # self.msg2.data = '{"action":"2","steerAngle":'+str(0.0)+'}'
        # self.cmd_vel_pub.publish(self.msg)
        # self.cmd_vel_pub.publish(self.msg2)
    def go_back(self):
        # self.cmd_vel_pub(0.0, -0.2)
        self.msg.data = '{"action":"1","speed":'+str(-0.2)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(0.0)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)

    #helper functions
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
        # Calculate the steering angle
        image_center = 640 / 2

        if center==image_center:
            center=self.center
            self.center=image_center
        else:
            self.center=center
        
        error = (center - image_center)
        d_error = error-self.last
        self.last = error
        steering_angle = (error * self.p+d_error*self.d)
        steering_angle = np.clip(steering_angle, -0.4, 0.4)

        return steering_angle

    def publish_cmd_vel(self, steering_angle, velocity = None):
        """
        Publish the steering command to the cmd_vel topic
        :param steering_angle: Steering angle in radians
        """
        # if velocity is None:
        #     velocity = self.maxspeed/2 + (self.maxspeed/2)*abs(steering_angle)/0.4
        # self.msg.data = '{"action":"1","speed":'+str(velocity)+'}'
        self.msg.data = '{"action":"1","speed":'+str(self.maxspeed)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(steering_angle*180/np.pi)+'}'
        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)

if __name__ == '__main__':
    node = LaneFollower()
    while not rospy.is_shutdown():
        node.rate.sleep()
        rospy.spin()
