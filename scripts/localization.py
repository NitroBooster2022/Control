#!/usr/bin/env python3

import sys
import rospy
import cv2
import os
import numpy as np
from utils.msg import localisation, IMU
from pynput import keyboard
from queue import Queue
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from trackmap import track_map
from utils.srv import *

class Localiser():
    def __init__(self):
        self.position = Queue(maxsize = 5)
        self.position.put(np.array([0,0]))
        self.position.put(np.array([0,0]))
        self.position.put(np.array([0,0]))
        self.position.put(np.array([0,0]))
        self.position.put(np.array([0,0]))
        self.pos = np.array([1,15])
        self.rot = 0
        self.map = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/templates/map.png')
        self.map_history = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/templates/map.png')

        # planned path
        # self.planned_path=['int1E','int2N','int5N','int6E','int6S','int4W','int3W','int1S','start']
        # self.planned_path=['int1N','int3N','track1N','parkingN']
        self.planned_path=['parkingN','roundabout','int5N','int1E','roundabout','parkingN']
        self.path_number=0
        self.track_map=track_map(self.pos[0],self.pos[1],self.rot,[self.planned_path[self.path_number]])
        self.planned_path=self.track_map.path

        # windowName = "Map"
        # cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(windowName,700,700)

        rospy.init_node('localisation_node', anonymous=True)
        self.localisation_sub = rospy.Subscriber("/automobile/localisation", localisation, self.callback)
        self.server = rospy.Service("get_direction", get_direction, self.doDir)
        self.rate = rospy.Rate(5)
        # self.keyInput()

    def doDir(self,request):
        # get the next direction
        if request.req == 'direction':
            if len(self.planned_path)==self.path_number+1:
                return 'idle'
            # self.track_map=track_map(request.x,request.y,request.r,[request.dest])
            self.track_map.locate(self.pos[0],self.pos[1],self.rot)
            self.track_map.planned_path=[self.planned_path[self.path_number]]
            self.track_map.plan_path()
            if self.track_map.path[0]==self.planned_path[self.path_number]:
                self.path_number+=1
                self.track_map.locate(self.pos[0],self.pos[1],self.rot)
                self.track_map.planned_path=[self.planned_path[self.path_number]]
                self.track_map.plan_path()
            response = self.track_map.directions[0]
            if self.track_map.decision_point(self.pos[0],self.pos[1])==False:
                response = 'straight'
            return response
        elif request.req == 'offset':
            # offset calculation
            orientations = np.array([0,np.pi/2,np.pi,-np.pi/2,-np.pi])
            dif = np.absolute(orientations-self.rot)
            offset = self.rot-orientations[dif.argmin()]
            response = str(offset)
            return response
        elif request.req == 'orientation':
            # orientation calculation (for traffic lights)
            if dif.argmin()==0:
                response = 'E'
            elif dif.argmin()==1:
                response = 'N'
            elif dif.argmin()==3:
                response = 'S'
            else:
                response = 'W'
            # rospy.loginfo("direction is: %s", response)
            return response
        else:
            return ''
    
    def callback(self,data):
        # update position and orientation
        self.position.get()
        self.position.put(np.array([data.posA,data.posB]))

        positions = []
        for i in range(self.position.maxsize):
            p=self.position.get()
            positions.append(p)
            self.position.put(p)
        self.pos = sum(positions)/len(positions)
        self.rot = data.rotA

        # uncomment to show location on map
        # self.show_map()

    def show_map(self):
        # display on map
        pos2 = self.pos+np.array([0.2*np.cos(-self.rot),0.2*np.sin(-self.rot)])
        cv2.circle(self.map_history, (int(self.pos[0]/15*self.map.shape[0]),int(self.pos[1]/15*self.map.shape[0])), radius=10, color=(0,0,255), thickness=-1)
        img_map = np.copy(self.map)
        img_map = cv2.arrowedLine(img_map, (int(self.pos[0]/15*self.map.shape[0]),int(self.pos[1]/15*self.map.shape[0])),
        (int(pos2[0]/15*self.map.shape[0]),int(pos2[1]/15*self.map.shape[0])), color=(0,0,255), thickness=10)
        cv2.circle(img_map, (int(self.pos[0]/15*self.map.shape[0]),int(self.pos[1]/15*self.map.shape[0])), radius=20, color=(0,0,255), thickness=-1)
        cv2.imshow("Map", img_map)
        key = cv2.waitKey(1)

    def keyInput(self):
        self.allKeys = ['p','m']
        with keyboard.Listener(on_press = self.keyPress) as listener:
            listener.join()

    # ===================================== KEY PRESS ====================================
    def keyPress(self,key):
        """Processing the key pressing 
        
        Parameters
        ----------
        key : pynput.keyboard.Key
            The key pressed
        """                                     
        try:
            if key.char in self.allKeys:
                if key.char == 'p':
                    # print the path taken
                    windowName = "path"
                    cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(windowName,500,500)
                    cv2.imshow(windowName, self.map_history)
                    key = cv2.waitKey(1)
                elif key.char == 'm':
                    # print the planned path
                    self.track_map.locate(self.pos[0],self.pos[1],self.rot)
                    self.track_map.planned_path=self.planned_path
                    self.track_map.plan_path()
                    self.track_map.draw_map()
        except: pass

if __name__ == '__main__':
    try:
        node = Localiser()
        node.rate.sleep()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass