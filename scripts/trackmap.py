#!/usr/bin/env python3

import networkx as nx
import numpy as np
import cv2
import os
import json

class track_map():

    def __init__(self,posX=0,posY=0,rot=0,path=[]):
        self.map = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/templates/map.png')
        self.map_graph = nx.DiGraph()
        self.locations = ['start','int1N','int1S','int1W','int1E',
        'int2N','int2S','int2W','int2E','int3N','int3S','int3W','int3E',
        'int4N','int4S','int4W','int4E','int5N','int5W','int5E',
        'int6N','int6S','int6W','int6E','track1N','track1S','track2N',
        'track2S','track3N','track3S','parkingN',
        'parkingS','roundabout','highwayN','highwayS','curvedpath']
        self.locations_coord = [[0.75,14.25],[0.75,12.25],[0.75,12.25],
        [1.5,13.75],[1.5,13.75],[3,12.25],[3,12.25],[4,13.75],[4,13.75],
        [0.75,9],[0.75,9],[1.5,10.75],[1.5,10.75],[5,12.25],[5,12.25],
        [4,10.75],[4,10.75],[3,9],[1.5,7.25],[1.5,7.25],[5,9],[5,9],
        [4,7.25],[4,7.25],[1.5,4],[1.5,4],[9.5,2.25],[9.5,2.25],[6.5,5.5],
        [6.5,5.5],[4,2.25],[4,2.25],[9.5,4],[10,10],[10,10],[12,12]]
        self.location_decision_points = [[0.83,14.53],[0.83,11.63],
        [0.45,12.56],[1.63,13.32],[1.93,13.71],
        [3.05,11.63],[2.67,12.56],[3.83,13.32],[4.13,13.71],
        [0.83,7.83],[0.45,9.69],[1.63,10.44],[1.93,10.83],
        [5.28,11.63],[4.9,12.56],[3.83,10.44],[4.13,10.83],
        [3.05,7.83],[1.63,6.61],[1.93,7],
        [5.28,7.83],[4.9,9.69],[3.83,6.61],[4.13,7],
        [2.7,2],[0.45,5.83],[11,3.5],[4.6,1.7],[4.9,5.87],[8.56,3.92],
        [4.3,2.1],[3,1.7],[0,0],[10,5],[6,13],[11.9,4.7]]
        self.dashedlines = ['parkingN','parkingS','track2N','track2S',
        'int5N','int6N','int6S','highwayN','highwayS']
        self.planned_path = path
        
        # graph map creation (run to get edgelist)
        # self.make_map()

        # loading the graph map

        self.map_graph=nx.read_edgelist(os.path.dirname(os.path.realpath(__file__))+'/templates/map.edgelist',create_using=nx.DiGraph())
        for i in range(len(self.map_graph.nodes)):
            self.map_graph.nodes[self.locations[i]]['coord']=self.locations_coord[i]
        # self.map_graph=nx.read_graphml(os.path.dirname(os.path.realpath(__file__))+'/templates/Competition_track.graphml')

        # get the current location
        self.location = self.locate(posX,posY,rot)

        # self.custum_path()

        # calculate the shortest path
        self.path = []
        self.directions = []
        # self.plan_path()

        # Convert to undirected graph and compute shortest paths
        # H = self.map_graph.to_undirected()

        # # Use TSP algorithm to find shortest path
        # tsp_path = nx.approximation.traveling_salesman_problem(H,weight="weight",nodes=H.nodes)

        # # Convert TSP solution to directed path
        # directed_path = []
        # for i in range(len(tsp_path)-1):
        #     u = tsp_path[i]
        #     v = tsp_path[i+1]
        #     directed_path += nx.shortest_path(self.map_graph, u, v)

        # print(directed_path)
        # print(len(directed_path))
        # self.path = directed_path

        # Check if all edges have been traversed
        # all_edges = set(self.map_graph.edges())
        # traversed_edges = set(zip(directed_path, directed_path[1:]))
        # untraversed_edges = all_edges - traversed_edges
        # # Print untraversed edges
        # if untraversed_edges:
        #     print("The following edges were not traversed:")
        #     for u, v in untraversed_edges:
        #         print(f"{u} -> {v}")
        # else:
        #     print("All edges have been traversed.")

        # path = []
        # for u, v in untraversed_edges:
        #     shortest_path = nx.shortest_path(self.map_graph, u, v, weight='weight')
        #     path += shortest_path[:-1]
        # # path += [untraversed_edges[-1][-1]]

        # print(path)

        # traversed_edges = set(zip(path, path[1:]))
        # untraversed_edges = untraversed_edges - traversed_edges
        # # Print untraversed edges
        # if untraversed_edges:
        #     print("The following edges were not traversed:")
        #     for u, v in untraversed_edges:
        #         print(f"{u} -> {v}")
        # else:
        #     print("All edges have been traversed.")

        # Save directed path to a JSON file
        # with open('/home/antoinedeng/Documents/Simulator/src/control/scripts/directed_path.json', 'w') as outfile:
        #     json.dump(directed_path, outfile)

    def get_location_dest(self,loc):
        dirs = []
        # print(self.map_graph.out_edges(loc,data=True))
        for _, dest, data in self.map_graph.out_edges(loc, data=True):
            dirs.append(data['dir'])
        # print(dirs)
        return dirs

    def get_location_cood(self,loc):
        return self.map_graph.nodes[loc]['coord']

    def custum_path(self, save=False, path="/paths/savedpaths/path.json"):
        print("---Click on map to input path---")
        print("---Press any keys to continue---")
        self.regions = cv2.imread(os.path.dirname(os.path.realpath(__file__))+'/templates/map_graphv2.drawio.png')
        self.planned_path = []
        windowName = 'path'
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,700,700)
        cv2.setMouseCallback(windowName,self.mouse_event_handler)
        cv2.imshow(windowName,self.regions)
        key = cv2.waitKey(0)
        # self.plan_path()
        if save:
            # Save directed path to a JSON file
            with open(os.path.dirname(os.path.realpath(__file__))+path, 'w') as outfile:
                json.dump(self.planned_path, outfile)
        cv2.destroyAllWindows()

    def mouse_event_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"User clicked on ({x}, {y})")
            print(self.locateM(x,y))
            self.planned_path.append(self.locateM(x,y))

    def plan_path(self):
        # get the path and directions
        self.path = [self.location]
        loc = self.location
        for i in range(len(self.planned_path)):
            if self.planned_path[i] != "start": #remove edges going to start
                try:
                    self.rm_edge('int1S','start')
                    self.rm_edge('int1W','start')
                except:
                    pass
            else:
                self.add_edge('int1S','start',1)
                self.add_edge('int1W','start',0)
            if self.planned_path[i] != "parkingN" and self.planned_path[i] != "parkingS" and loc != "parkingN" and loc != "parkingS":
                self.rm_edge('track1N','parkingN')
                self.rm_edge('parkingN','track2N')
                self.rm_edge('parkingN','track1S')
                self.rm_edge('parkingS','track1S')
                self.rm_edge('track2S','parkingS')
            p = nx.shortest_path(self.map_graph, source=loc, target=self.planned_path[i], weight='weight')
            loc = self.planned_path[i]
            self.path += p[1:]
            if self.planned_path[i] != "parkingN" and self.planned_path[i] != "parkingS" and loc != "parkingN" and loc != "parkingS":
                self.add_edge('track1N','parkingN',3)
                self.add_edge('parkingN','track2N',5)
                self.add_edge('parkingN','track1S',6)
                self.add_edge('parkingS','track1S',7)
                self.add_edge('track2S','parkingS',4)
        self.directions = []
        sc = []
        for i in range(len(self.path)-1):
            d=self.map_graph.get_edge_data(self.path[i],self.path[i+1]).get('dir')
            if d != -1:
                self.directions.append(d)
            else:
                sc.append(i)
        for i in reversed(sc):
            self.path.pop(i)
        print("planned path:")
        print(self.path)
        print("direction list:")
        print(self.directions)

    def draw_map_graphml(self):
        # draw the path (graphml)
        self.map_graph=nx.read_graphml(os.path.dirname(os.path.realpath(__file__))+'/templates/Competition_track.graphml')
        img_map=self.map
        # cv2.circle(img_map, (int(self.map_graph.nodes[self.location]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[self.location]['coord'][1]/15*self.map.shape[1])), radius=20, color=(0,255,0), thickness=-1)
        # print(len(self.map_graph.nodes)-1)
        # for i in range(len(self.map_graph.nodes)):
        #     try:
        #         cv2.circle(img_map, (int(float(self.map_graph.nodes[str(i)]['x'])/15*self.map.shape[0]),int(float(self.map_graph.nodes[str(i)]['y'])/15*self.map.shape[1])), radius=15, color=(0,255,0), thickness=-1)
        #     except:
        #         pass
        # for i in range(len(self.map_graph.edges)):
        #     try:
        #         img_map = cv2.arrowedLine(img_map, (int(float(self.map_graph.nodes[str(i)]['x'])/15*self.map.shape[0]),int(float(self.map_graph.nodes[str(i)]['y'])/15*self.map.shape[1])),
        #             ((int(float(self.map_graph.nodes[str(i+1)]['x'])/15*self.map.shape[0]),int(float(self.map_graph.nodes[str(i+1)]['y'])/15*self.map.shape[1]))), color=(255,0,255), thickness=10)
        #     except:
        #         pass
        for n in self.map_graph.nodes():
            cv2.circle(img_map, (int(float(self.map_graph.nodes[n]['x'])/15*self.map.shape[0]),int(float(self.map_graph.nodes[n]['y'])/15*self.map.shape[1])), radius=15, color=(0,255,0), thickness=-1)
        for e in self.map_graph.edges():
            source = e[0]
            dest = e[1]
            img_map = cv2.arrowedLine(img_map, (int(self.map_graph.nodes[source]['x']/15*self.map.shape[0]),int(self.map_graph.nodes[source]['y']/15*self.map.shape[1])),
                    ((int(self.map_graph.nodes[dest]['x']/15*self.map.shape[0]),int(self.map_graph.nodes[dest]['y']/15*self.map.shape[1]))), color=(255,0,255), thickness=5)
        windowName = 'track'
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,700,700)
        cv2.imshow(windowName, img_map)
        key = cv2.waitKey(0)

    def draw_map_edgelist(self):
        # draw the path (edgelist)
        img_map=self.map
        for i in range(len(self.map_graph.nodes)):
            cv2.putText(img_map, self.locations[i], (int(self.map_graph.nodes[self.locations[i]]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[self.locations[i]]['coord'][1]/15*self.map.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 3, cv2.LINE_AA)
        for e in self.map_graph.edges():
            source = e[0]
            # print(type(source))
            dest = e[1]
            img_map = cv2.arrowedLine(img_map, (int(self.map_graph.nodes[source]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[source]['coord'][1]/15*self.map.shape[1])),
                    ((int(self.map_graph.nodes[dest]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[dest]['coord'][1]/15*self.map.shape[1]))), color=(255,0,255), thickness=10)
        windowName = 'track'
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,700,700)
        cv2.imshow(windowName, img_map)
        key = cv2.waitKey(0)

    def draw_map(self):
        # draw the path
        img_map=self.map
        self.draw_error(1,1,img_map)
        cv2.circle(img_map, (int(self.map_graph.nodes[self.location]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[self.location]['coord'][1]/15*self.map.shape[1])), radius=20, color=(0,255,0), thickness=-1)
        for i in range(len(self.map_graph.nodes)):
            cv2.putText(img_map, self.locations[i], (int(self.map_graph.nodes[self.locations[i]]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[self.locations[i]]['coord'][1]/15*self.map.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 3, cv2.LINE_AA)
        for i in range(len(self.path)-1):
            img_map = cv2.arrowedLine(img_map, (int(self.map_graph.nodes[self.path[i]]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[self.path[i]]['coord'][1]/15*self.map.shape[1])),
            (int(self.map_graph.nodes[self.path[i+1]]['coord'][0]/15*self.map.shape[0]),int(self.map_graph.nodes[self.path[i+1]]['coord'][1]/15*self.map.shape[1])), color=(255,0,255), thickness=10)
        windowName = 'track'
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName,700,700)
        cv2.imshow(windowName, img_map)
        key = cv2.waitKey(0)

    def draw_error(self, x, y, image):
        cv2.circle(image, (int(x/15*self.map.shape[0]),int(y/15*self.map.shape[1])), radius=int(0.2/15*self.map.shape[1]), color=(0,255,0), thickness=-1)

    def decision_point(self,x,y): # include yaw check as well
        loc = self.location_decision_points[self.locations.index(self.location)]
        d = (loc[0]-x)**2+(loc[1]-y)**2
        if d<0.25:
            return True
        else:
            return False

    def can_overtake(self,x,y,rot):
        return self.locate(x,y,rot) in self.dashedlines

    def locate(self,x,y,rot):
        # get current location based on position and orientation
        if rot > np.pi:
            rot -= 2*np.pi
        if y>6.2:
            if x<1.3:
                if y<10.75:
                    if rot<np.pi and rot>0:
                        return 'int3N'
                    else:
                        return 'int3S'
                elif y<13.7:
                    if rot<np.pi and rot>0:
                        return 'int1N'
                    else:
                        return 'int1S'
                else:
                    return 'start'
            elif x<2.55:
                if y<8.75:
                    if rot<np.pi/2 and rot>-np.pi/2:
                        return 'int5E'
                    else:
                        return 'int5W'
                elif y<12.25:
                    if rot<np.pi/2 and rot>-np.pi/2:
                        return 'int3E'
                    else:
                        return 'int3W'
                else:
                    if rot<np.pi/2 and rot>-np.pi/2:
                        return 'int1E'
                    else:
                        return 'int1W'
            elif x<3.25:
                if y<10.75:
                    # no int5S
                    return 'int5N'
                else:
                    if rot<np.pi and rot>0:
                        return 'int2N'
                    else:
                        return 'int2S'
            elif x<4.65:
                if y<8.75:
                    if rot<np.pi/2 and rot>-np.pi/2:
                        return 'int6E'
                    else:
                        return 'int6W'
                elif y<12.25:
                    if rot<np.pi/2 and rot>-np.pi/2:
                        return 'int4E'
                    else:
                        return 'int4W'
                else:
                    if rot<np.pi/2 and rot>-np.pi/2:
                        return 'int2E'
                    else:
                        return 'int2W'
            elif x<5.5:
                if y<10.75:
                    if rot<np.pi and rot>0:
                        return 'int6N'
                    else:
                        return 'int6S'
                else:
                    if rot<np.pi and rot>0:
                        return 'int4N'
                    else:
                        return 'int4S'
            elif x>11:
                return 'curvedpath'
            else:
                if rot<3/4*np.pi and rot>-1/4*np.pi:
                    return 'highwayN'
                else:
                    return 'highwayS'
        else:
            if x<2.75:
                if rot<3/4*np.pi and rot>-1/4*np.pi:
                    return 'track1N'
                else:
                    return 'track1S'
            elif x<4.75:
                if rot<3/4*np.pi and rot>-1/4*np.pi:
                    return 'parkingN'
                else:
                    return 'parkingS'
            elif y<2.9:
                if x<13:
                    if rot<3/4*np.pi and rot>-1/4*np.pi:
                        return 'track2N'
                    else:
                        return 'track2S'
                else:
                    if rot<1/4*np.pi and rot>-3/4*np.pi:
                        return 'track2N'
                    else:
                        return 'track2S'
            elif x<8.6:
                if rot<3/4*np.pi and rot>-1/4*np.pi:
                    return 'track3N'
                else:
                    return 'track3S'
            elif y<4.6:
                if x<10.75:
                    return 'roundabout'
                else:
                    if rot<1/2*np.pi and rot>-1/2*np.pi:
                        return 'track2S'
                    else:
                        return 'track2N'
            elif x>11:
                return 'curvedpath'
            else:
                if rot<3/4*np.pi and rot>-1/4*np.pi:
                    return 'highwayN'
                else:
                    return 'highwayS'
    
    def locateM(self,x,y):
        if y>1030:
            if x<185:
                if y<1640:
                    if x>125:
                        return 'int3N'
                    else:
                        return 'int3S'
                elif y<2050:
                    if x>125:
                        return 'int1N'
                    else:
                        return 'int1S'
                else:
                    return 'start'
            elif x<400:
                if y<1155:
                    if y>1100:
                        return 'int5E'
                    else:
                        return 'int5W'
                elif y<1730:
                    if y>1680:
                        return 'int3E'
                    else:
                        return 'int3W'
                else:
                    if y>2120:
                        return 'int1E'
                    else:
                        return 'int1W'
            elif x<525:
                if y<1630:
                    # no int5S
                    return 'int5N'
                else:
                    if x>465:
                        return 'int2N'
                    else:
                        return 'int2S'
            elif x<745:
                if y<1155:
                    if y>1100:
                        return 'int6E'
                    else:
                        return 'int6W'
                elif y<1735:
                    if y>1685:
                        return 'int4E'
                    else:
                        return 'int4W'
                else:
                    if y>2120:
                        return 'int2E'
                    else:
                        return 'int2W'
            elif x<860:
                if y<1640:
                    if x>795:
                        return 'int6N'
                    else:
                        return 'int6S'
                else:
                    if x>795:
                        return 'int4N'
                    else:
                        return 'int4S'
            elif x>1750:
                return 'curvedpath'
            else:
                if x>1550:
                    return 'highwayN'
                else:
                    return 'highwayS'
        else:
            if x<435:
                if x>245:
                    return 'track1N'
                else:
                    return 'track1S'
            elif x<710:
                if y>355:
                    return 'parkingN'
                else:
                    return 'parkingS'
            elif y<465:
                if y>300:
                    return 'track2N'
                else:
                    return 'track2S'
            elif x<1340:
                if x>1035:
                    return 'track3N'
                else:
                    return 'track3S'
            else:
                return 'roundabout'

    # graph creation helpers
    def add_edge(self,source,dest,d):
        w = (abs(self.map_graph.nodes[source]['coord'][0]-self.map_graph.nodes[dest]['coord'][0])+
        abs(self.map_graph.nodes[source]['coord'][1]-self.map_graph.nodes[dest]['coord'][1]))
        self.map_graph.add_edge(source,dest,weight=w,dir=d)

    def rm_edge(self,source,dest):
        self.map_graph.remove_edge(source,dest)
    
    def add_all_edges(self):
        #0:left, 1:straight, 2:right, 3:parkF, 4:parkP, 5:exitparkL, 6:exitparkR, 7:exitparkP
        #8:enterhwLeft, 9:enterhwStright, 10:rdb, 11:exitrdbE, 12:exitrdbS, 13:exitrdbW, 14:curvedpath
        self.add_edge('start','int1N',1)
        self.add_edge('start','int1E',2)
        self.add_edge('int1N','int3N',1)
        self.add_edge('int1N','int3E',2)
        self.add_edge('int1S','start',1)
        self.add_edge('int1S','int1E',0)
        self.add_edge('int1W','start',0)
        self.add_edge('int1W','int1N',2)
        self.add_edge('int1E','int2E',1)
        self.add_edge('int1E','int2N',0)
        self.add_edge('int2N','int5N',1)
        self.add_edge('int2N','int3W',0)
        self.add_edge('int2N','int4E',2)
        self.add_edge('int2S','int1W',2)
        self.add_edge('int2S','int2E',0)
        self.add_edge('int2W','int1W',1)
        self.add_edge('int2W','int2N',2)
        self.add_edge('int2E','int4N',0)
        self.add_edge('int2E','highwayN',9)
        self.add_edge('int3N','track1N',1)
        self.add_edge('int3N','int5E',2)
        self.add_edge('int3S','int1S',1)
        self.add_edge('int3S','int3E',0)
        self.add_edge('int3W','int1S',0)
        self.add_edge('int3W','int3N',2)
        self.add_edge('int3E','int5N',0)
        self.add_edge('int3E','int2S',2)
        self.add_edge('int3E','int4E',1)
        self.add_edge('int4N','int6N',1)
        self.add_edge('int4N','int4W',0)
        self.add_edge('int4S','int2W',2)
        self.add_edge('int4S','highwayN',8)
        self.add_edge('int4W','int2S',0)
        self.add_edge('int4W','int5N',2)
        self.add_edge('int4W','int3W',1)
        self.add_edge('int4E','int4S',2)
        self.add_edge('int4E','int6N',0)
        self.add_edge('int5N','int6E',2)
        self.add_edge('int5N','int5W',0)
        self.add_edge('int5W','track1N',2)
        self.add_edge('int5W','int3S',0)
        self.add_edge('int5E','int6E',1)
        self.add_edge('int6N','int6W',0)
        self.add_edge('int6N','track3N',1)
        self.add_edge('int6S','int4W',2)
        self.add_edge('int6S','int4S',1)
        self.add_edge('int6W','int5W',1)
        self.add_edge('int6E','int6S',2)
        self.add_edge('int6E','track3N',0)
        self.add_edge('track1N','parkingN',3)
        self.add_edge('track1N','track2N',-1)
        self.add_edge('track1S','int5E',0)
        self.add_edge('track1S','int3S',1)
        self.add_edge('parkingN','track2N',5)
        self.add_edge('parkingN','track1S',6)
        self.add_edge('parkingS','track1S',7)
        self.add_edge('track2S','parkingS',4)
        self.add_edge('track2S','track1S',-1)
        self.add_edge('track2N','roundabout',10)
        self.add_edge('roundabout','track2S',11)
        self.add_edge('roundabout','highwayS',12)
        self.add_edge('roundabout','track3S',13)
        self.add_edge('track3N','roundabout',10)
        self.add_edge('track3S','int6W',2)
        self.add_edge('track3S','int6S',1)
        self.add_edge('highwayN','roundabout',10)
        self.add_edge('highwayS','int4N',2)
        self.add_edge('highwayS','int2W',1)
        self.add_edge('curvedpath','track2S',2)
        self.add_edge('curvedpath','roundabout',0)
        self.add_edge('highwayN','curvedpath',14)

    def make_map(self):
        self.map_graph.add_nodes_from(self.locations)
        for i in range(len(self.locations_coord)):
            self.map_graph.nodes[self.locations[i]]['coord']=self.locations_coord[i]
        self.add_all_edges()
        nx.write_edgelist(self.map_graph,os.path.dirname(os.path.realpath(__file__))+'/templates/map.edgelist')

if __name__ == '__main__':
    m = ['int1E','int2E','curvedpath','track2S','track1S','int6E','int6S']
    # m = json.load(open(os.path.dirname(os.path.realpath(__file__))+'/paths/path.json', 'r'))
    # print(m)
    node = track_map(0,15,1.5,m)
    # node.get_location_dest('start')
    # node.make_map()
    # node.draw_map()
    # node.custum_path()
    # node.custum_path(save=True)
    node.plan_path()
    node.draw_map()
    # node.draw_map_edgelist()
    node.draw_map_graphml()