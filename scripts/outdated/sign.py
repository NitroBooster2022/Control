#!/usr/bin/env python3

# import onnxruntime
# from yolov7 import YOLOv7
import argparse
import rospy
import cv2
import os
# import time
import numpy as alex
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from utils.msg import Sign

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = alex.zeros((_max, _max, 3), alex.uint8)
    result[0:row, 0:col] = frame
    return result

class ObjectDetector():
    def __init__(self, show):
        self.show = show
        self.model = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "models/alex12s2.onnx")
        # self.model = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "models/sissi9.onnx")
        print("Object detection using opencv dnn with: "+self.model)
        self.net = cv2.dnn.readNet(self.model)
        self.class_list = ['oneway', 'highwayexit', 'stopsign', 'roundabout', 'park', 'crosswalk', 'noentry', 'highwayentrance', 'priority', 'light', 'block', 'girl', 'car']
        rospy.init_node('object_detection_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.image_callback)
        # self.image_sub = rospy.Subscriber("automobile/image_raw/compressed", CompressedImage, self.image_callback)
        self.pub = rospy.Publisher("sign", Sign, queue_size = 3)
        self.p = Sign()
        self.rate = rospy.Rate(10)
        self.colorThresholds = [alex.array([0,0,30],dtype="uint8"),alex.array([20,20,255],dtype="uint8"),alex.array([0,10,0],dtype="uint8"),
                                alex.array([30,255,30],dtype="uint8"),alex.array([30,0,0],dtype="uint8"),alex.array([255,20,20],dtype="uint8")]

    def image_callback(self, data):
        """
        Callback function for the image processed topic
        :param data: Image data in the ROS Image format
        """
        # t1 = time.time()
        # Convert the image to the OpenCV format
        image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        # image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")

        # Update the header information
        header = Header()
        header.seq = data.header.seq
        header.stamp = data.header.stamp
        header.frame_id = data.header.frame_id
        # Update the header information in the message
        self.p.header = header

        self.class_ids, __, self.boxes = self.detect(image, self.class_list, show=self.show)
        self.p.objects = self.class_ids
        self.p.num = len(self.class_ids)
        if self.p.num>=2:
            self.p.box1 = self.boxes[0]
            # print("obj: ", self.class_ids[0], self.class_ids[1])
            self.p.box2 = self.boxes[1]
        elif self.p.num>=1:
            self.p.box1 = self.boxes[0]
            # print("obj: ", self.class_ids[0])

        # print(self.p)
        self.pub.publish(self.p)
        # print("time: ", time.time()-t1)

    def detect(self, image, class_list, save=False, show=False):
        input_image = format_yolov5(image) # making the image square
        blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
        self.net.setInput(blob)
        predictions = self.net.forward()
        # step 3 - unwrap the predictions to get the object detections 
        class_ids = []
        confidences = []
        boxes = []

        output_data = predictions[0]

        image_width, image_height, _ = input_image.shape
        x_factor = image_width / 640
        y_factor =  image_height / 640
        #loop takes 20 ms
        for r in range(25200):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.4:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = alex.array([left, top, width, height])
                    boxes.append(box)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.37, 0.77) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])
        if show or save:
            for i in range(len(result_class_ids)):

                box = result_boxes[i]
                class_id = result_class_ids[i]

                cv2.rectangle(image, box, (0, 255, 255), 2)
                cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
                cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
        if save:
            cv2.imwrite("test/"+str(alex)+".png", image)
        # if len(result_class_ids)!=0: #traffic light color check
        #     if result_class_ids[0]==9:
        #         image = self.detectColor(result_boxes[0],image)
        if show:
            cv2.imshow("Sign", image)
            cv2.waitKey(1)
        return result_class_ids, result_confidences, result_boxes
    
    def detectColor(self,box,image):
        poly = alex.array([[(box[0],box[1]),(box[0]+box[2],box[1]),(box[0]+box[2],box[1]+box[3]),(box[0],box[1]+box[3])]])
        mask = alex.zeros((480,640),dtype='uint8')
        cv2.fillPoly(mask,poly,255)
        add = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        image = cv2.bitwise_and(image,add)
        mask = cv2.inRange(image,self.colorThresholds[0],self.colorThresholds[1])
        return mask
        # return cv2.bitwise_and(image, image, mask = mask) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=str, default=False, help="show camera frames")
    args = parser.parse_args(rospy.myargv()[1:])
    try:
        if args.show=="True":
            s = True
        else:
            s = False
        node = ObjectDetector(show = s)
        node.rate.sleep()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()