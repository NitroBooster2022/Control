#!/usr/bin/env python3

import argparse
import rospy
import cv2
import os
import time 
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
# from pynput import keyboard
from std_msgs.msg import Header
from utils.msg import Sign, Light
import tensorrt as trt
import onnxruntime



# -------------------helpers----------------------------------------------------
#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from itertools import chain
import argparse
import os

import pycuda.driver as cuda
# import pycuda.autoinit
cuda.init()
device = cuda.Device(0)
ctx = device.make_context()
import numpy as np

import tensorrt as trt

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "usr", "src", "tensorrt", "data")
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory, and any additional data directories.", action="append", default=[kDEFAULT_DATA_ROOT])
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            print("WARNING: " + data_path + " does not exist. Trying " + data_dir + " instead.")
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)):
            print("WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(data_path))
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files)

def locate_files(data_paths, filenames):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError("Could not find {:}. Searched in data paths: {:}".format(filename, data_paths))
    return found_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
# mmodified 02/02: show TRT inference using onnx - Error Code 1: Cuda Driver (invalid resource handle), cuda_ctx added
def do_inference_v2(cuda_ctx, context, bindings, inputs, outputs, stream):
    cuda_ctx.push()
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    cuda_ctx.pop()
    # Return only the host outputs.
    return [out.host for out in outputs]

def getColor(img,img_hsv,color):
    
   
    lower_red = np.array([170,80,80])
    upper_red = np.array([179,255,255])
    
    lower_green = np.array([40,80,80])
    upper_green = np.array([90,255,255])
    
    lower_yellow = np.array([25,80,80])
    upper_yellow = np.array([35,255,255])
    
    # Threshold the HSV image to get only rgb colors
    if color.__eq__("red"):
        mask = cv2.inRange(img_hsv, lower_red, upper_red) + cv2.inRange(img_hsv, np.array([0,80,80]), np.array([20,255,255]))
    elif color.__eq__("green"):
        mask = cv2.inRange(img_hsv, lower_green, upper_green)
    elif color.__eq__("yellow"):
        mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    else:
        print("no such color, use red/green/yellow")
        return
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    mask_1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the colored areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)   
    largest = 10
    crop_contour = None
    for contour in contours:
        
        area = cv2.contourArea(contour)
        # print(area)
        if area > largest:
            largest = area
            # print(largest)
            crop_contour = contour
    if largest > 10:
        height,width = cv2.split(crop_contour)
        crop_hmax = np.max(height)
        crop_hmin = np.min(height)
        crop_wmax = np.max(width)
        crop_wmin = np.min(width)
        
        
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img_hsv,img_hsv, mask = mask_1)
        crop_res = res[crop_wmin:crop_wmax,crop_hmin:crop_hmax]
        # crop_black = np.delete(crop_res,np.where(crop_res == [0,0,0]),axis=None)
        _,s,v = cv2.split(crop_res)
        
        # h = np.delete(h,np.where(v <= 20),None)
        # hue = np.mean(h)
        
        s = np.delete(s,np.where(v <= 80),None)
        v = np.delete(v,np.where(v <= 80),None)
        
        res = cv2.bitwise_and(img,img, mask = mask_1)
        crop_img = res[crop_wmin:crop_wmax,crop_hmin:crop_hmax]
        b,g,r = cv2.split(crop_img)
        
        b = b.flatten()
        g = g.flatten()
        r = r.flatten()
        
        if color=="red":
            r = np.delete(r,np.where(r <= 50),None)
            g = np.delete(g,np.where(r <= 50),None)
            b = np.delete(b,np.where(r <= 50),None)
        elif color=="green":
            r = np.delete(r,np.where(g <= 50),None)
            g = np.delete(g,np.where(g <= 50),None)
            b = np.delete(b,np.where(g <= 50),None)
        elif color=="yellow":
            r = np.delete(r,np.where(g <= 50) and np.where(r<=50),None)
            g = np.delete(g,np.where(g <= 50)and np.where(r<=50),None)
            b = np.delete(b,np.where(g <= 50)and np.where(r<=50),None)
        
        # print("ck1")
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)
        
        sat_mean = np.mean(s)
        val_mean = np.mean(v)

    else:
        b_mean = 0
        g_mean = 0
        r_mean = 0
        
        sat_mean = 0
        val_mean = 0
        
    return b_mean,g_mean,r_mean,sat_mean,val_mean
 
def predict(strR,strG,strY):
    
    max = np.max([strR,strG*1.05,strY])
    # print(max)
    if max == strR:
        return 1 #"red"
    elif max == strG*1.05:
        return 2 #"green"
    elif max == strY:
        return 2 #"green"
    else:
        return 3 #"undetermined"

def lightColor_identify(img):
    # t1 = time.time()
    img = cv2.GaussianBlur(img,(5,5),0)
    # img = img[int(w*0.2):int(w*0.8),int(h*0.2):int(h*0.8)]
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    b_red,g_red,r_red,sat_red,val_red = getColor(img,img_hsv,"red")
    b_green,g_green,r_green,sat_green,val_green = getColor(img,img_hsv,"green")
    b_yellow,g_yellow,r_yellow,sat_yellow,val_yellow = getColor(img,img_hsv,"yellow")

    red = 0
    green = 0
    yellow = 0
    if sat_red>0:
        red = (r_red+g_red)*val_red/sat_red
    if sat_green>0:
        green = (g_green+b_green)*val_green/sat_green 
    if sat_yellow>0:
        yellow = (r_yellow+g_yellow)/2*val_yellow/sat_yellow    
    
    print(red,green,yellow)
    light_color = predict(red,green,yellow)

    return light_color
    
# ----------------------------------------helpers end-------------------------------------------------------

global depth_img
confidence_thresholds = [0.8, 0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9]
distance_thresholds = [2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.2]
SIGN_H2D_RATIO = 31.57
LIGHT_W2D_RATIO = 41.87
CAR_H2D_RATIO = 90.15

class ObjectDetector():
    def __init__(self, show):
        self.show = show
        # self.model = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "models/np12s2.onnx")
        # self.model = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "models/sissi9s.onnx")
        # self.model = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "models/ningp10.onnx")
        self.model = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "models/best.engine")
        print("Object detection using tensorrt with: "+self.model)
        self.detector = InferenceModel(self.model, conf_thres=0.45, iou_thres=0.35)
        # self.net = cv2.dnn.readNet(self.model)
        self.class_names = ['oneway', 'highwayexit', 'stopsign', 'roundabout', 'park', 'crosswalk', 'noentry', 'highwayentrance', 'priority', 'light', 'block', 'girl', 'car']
        rospy.init_node('object_detection_node', anonymous=True)
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        # self.image_sub = rospy.Subscriber("automobile/image_raw/compressed", CompressedImage, self.image_callback)
        self.pub = rospy.Publisher("sign", Sign, queue_size = 3)
        self.light_pub = rospy.Publisher("light",Light, queue_size=3)
        self.p = Sign()
        self.light_color = Light()
        self.rate = rospy.Rate(15)


    def depth_callback(self, data):
        global depth_img
        depth_img = self.bridge.imgmsg_to_cv2(data,desired_encoding='passthrough')
    
    
    def image_callback(self, data):
        """
        Callback function for the image processed topic
        :param data: Image data in the ROS Image format
        """
        t1 = time.time()
        # Convert the image to the OpenCV format
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.show = True
         # Update the header information
        header = Header()
        header.seq = data.header.seq
        header.stamp = data.header.stamp
        header.frame_id = data.header.frame_id
        # Update the header information in the message
        self.p.header = header

        # self.class_ids, __, self.boxes = self.detect(image, self.class_list, show=self.show)
        self.boxes, self.scores, self.class_ids = self.detector(image)
        
        #detection result filtering 
        todelete = []
        self.distances=[]
        for i in range(len(self.scores)):
            if self.scores[i] >= confidence_thresholds[self.class_ids[i]]:
                distance = computeMedianDepth(depth_img,self.boxes[i])/1000
                if distance_make_sense(distance,self.class_ids[i],self.boxes[i]):
                    self.distances.append(distance)
                else:
                    todelete.append(i)
                
        self.boxes = [value for index, value in enumerate(self.boxes) if index not in todelete]
        self.scores = [value for index, value in enumerate(self.scores) if index not in todelete]
        self.class_ids = [value for index, value in enumerate(self.class_ids) if index not in todelete]
        self.distances = [value for index, value in enumerate(self.distances) if index not in todelete]
  
        #traffic light color detection
        color = 0
        if (9 in self.class_ids):
            indices = [index for index, element in enumerate(self.class_ids) if element == 9]
            closest_tl = 0
            tl_area = 0
            for index in indices:
                x1 = int(self.boxes[index][2])
                x2 = int(self.boxes[index][0])
                y1 = int(self.boxes[index][3])
                y2 = int(self.boxes[index][1])
                if tl_area < (x2-x1)*(y2-y1):
                    closest_tl = index
            x1 = int(self.boxes[closest_tl][2])
            x2 = int(self.boxes[closest_tl][0])
            y1 = int(self.boxes[closest_tl][3])
            y2 = int(self.boxes[closest_tl][1])
            # print(x1,x2,y1,y2)
            
            tlimg = image[y2:y1,x2:x1]
            color = lightColor_identify(tlimg)
        self.light_color.header = header
        self.light_color.light_color = color
        self.light_pub.publish(self.light_color)

        if self.show:
            img = draw_detections(image, self.boxes, self.scores, self.class_ids)
            # cv2.rectangle(image, (100, 100), (200, 300), (255,0,0), 2)
            cv2.imshow('sign', img)
            cv2.waitKey(3)
        self.p.objects = self.class_ids
        self.p.num = len(self.class_ids)
        self.p.confidence = self.scores
        self.p.distances = self.distances
        for i in range(len(self.boxes)):
            height1 = self.boxes[i][3]-self.boxes[i][1]
            width1 = self.boxes[i][2]-self.boxes[i][0]
            self.boxes[i][2] = width1
            self.boxes[i][3] = height1
            if i == 0:
                self.p.box1 = self.boxes[0]
            elif i==1:
                self.p.box2 = self.boxes[1]
            elif i==2:  
                self.p.box3 = self.boxes[2]
            elif i==3:
                self.p.box4 = self.boxes[3]
        
        self.pub.publish(self.p)
        print("time: ",time.time()-t1)

#detector class
class InferenceModel:

    def __init__(self, path, conf_thres=0.5, iou_thres=0.5, official_nms=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.official_nms = official_nms
        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)
    
    def initialize_model(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        
        #deserialize the engine from disk
        runtime = trt.Runtime(logger) 
        if os.path.exists(path):
            print("reading engine from file {}".format(path))
            with open(path,"rb") as f:
                serialized_engine = f.read() 
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            # self.engine = engine
        else:
            print("engine not found")
            exit(0)
        # print(type(self.engine))
        #inference
        self.context = self.engine.create_execution_context()
        
        #buffer pointer for input and ouputs
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        
        # self.trt_results = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        
        # self.session = onnxruntime.InferenceSession(path,
        #                                             providers=['CUDAExecutionProvider',
        #                                                        'CPUExecutionProvider'])
        # Get model info
        # self.get_input_details()
        self.get_io_details()
        # self.has_postprocess = 'score' in self.output_names or self.official_nms
        self.has_postprocess = self.official_nms

    def detect_objects(self, image):
        self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference()

        if self.has_postprocess:
            self.boxes, self.scores, self.class_ids = self.parse_processed_output(outputs)

        else:
            # Process output data
            self.boxes, self.scores, self.class_ids = self.new_process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255
        # HWC to CHW format:
        input_img = np.transpose(input_img, [2, 0, 1])
        # CHW to NCHW format
        input_img = np.expand_dims(input_img, axis=0)
        # Convert the image to row-major order, also known as "C order":
        input_img = np.array(input_img, dtype=np.float32, order="C")
        # Scale input pixel values to 0 to 1
        # input_img = input_img / 255.0
        # input_img = input_img.transpose(2, 0, 1)
        # input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        self.inputs[0].host = input_img
        # print(len(self.inputs))
        # return input_tensor 


    def inference(self):
        start = time.perf_counter()
        # self.inputs[0].host = input_tensor
        # self.outputs.clear()
        # self.bindings.clear()
        # print("ck1")
        trt_outputs = do_inference_v2(cuda_ctx=ctx,context=self.context,bindings=self.bindings,inputs = self.inputs,outputs=self.outputs,stream=self.stream)
        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        trt_outputs = [output.reshape(self.output_shapes[0][0]) for output in trt_outputs]
        return trt_outputs

    def new_process_output(self, outputs):
        outputs = np.array([cv2.transpose(outputs[0][0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                
                box = self.extract_boxes(outputs[0][i][0:4])
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
                
        # Apply NMS (Non-maximum suppression)
        result_boxes = (cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5))
        
        return [boxes[i] for i in result_boxes], [scores[i] for i in result_boxes], [class_ids[i] for i in result_boxes]

    

    def extract_boxes(self, box):
        

        # Scale boxes to original image dimensions
        box = self.rescale_boxes(box)

        # Convert boxes to xyxy format
        box = xywh2xyxy(box)

        return box

    def rescale_boxes(self, box):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        box = np.divide(box, input_shape, dtype=np.float32)
        box *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return box

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    # def get_input_details(self):
    #     # model_inputs = self.inputs[0].host
    #     # self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    #     self.input_shape = (640,640) #no explicit way to obtain the input shape from the engine, but we know it's 640x640 because the original onnx file is 640x640
    #     self.input_height = self.input_shape[0]
    #     self.input_width = self.input_shape[1]

    def get_io_details(self):
        self.output_shapes = []
        self.input_shapes = []
        for i in range(self.engine.num_bindings):
            if i%2==1: # we know that the output is always the second binding
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                self.output_shapes.append((shape, dtype))
                self.output_length = self.output_shapes[0][0][2]
            else: # we know that the input is always the first binding
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                self.input_shapes.append((shape, dtype))
                self.input_height = self.input_shapes[0][0][2]
                self.input_width = self.input_shapes[0][0][3]

#utils
class_names = ['oneway', 'highwayexit', 'stopsign', 'roundabout', 'park', 'crosswalk', 'noentry', 'highwayentrance', 'priority',
                'lights','block','pedestrian','car','others','others','others','others','others','others',
                'others','others','others','others','others','others','others','others','others','others',
                'others','others','others','others','others','others','others','others','others','others',
                'others','others','others','others','others','others','others','others','others','others',
                'others','others','others','others','others','others','others','others','others','others',
                'others','others','others','others','others','others','others','others','others','others',]

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = [int(item) for item in box]

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)


    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img

def computeMedianDepth(depthimg,box):
    x1 = int(max(0,box[0]))
    x2 = int(max(0,box[2]))
    y1 = int(max(0,box[1]))
    y2 = int(max(0,box[3]))
    croppedDepth = depthimg[y1:y2,x1:x2]
    depths = []
    depths = croppedDepth[croppedDepth>100].flatten().tolist()
    
    depths = sorted(depths)
    index20percent = len(depths)*0.2
    if (index20percent) <= 0:
        return depths[0]
    else:
        return depths[int(index20percent/2)]
    
def distance_make_sense(distance,objectID,box):
        if distance>distance_thresholds[objectID]:
            return False
        else:
            height = box[3]-box[1]
            width = box[2]-box[0]
            if objectID==12:
                expected_distance = CAR_H2D_RATIO/height
            elif objectID==9:
                expected_distance = LIGHT_W2D_RATIO/width
            elif objectID != 11:
                expected_distance = SIGN_H2D_RATIO/height
            if distance > expected_distance*1.33 or distance < expected_distance*1/1.33:
                return False
            else:
                return True

    
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

