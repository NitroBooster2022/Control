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
from utils.msg import Sign
import tensorrt as trt
import onnxruntime
import common

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
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.image_callback)
        # self.image_sub = rospy.Subscriber("automobile/image_raw/compressed", CompressedImage, self.image_callback)
        self.pub = rospy.Publisher("sign", Sign, queue_size = 3)
        self.p = Sign()
        self.rate = rospy.Rate(15)

    def image_callback(self, data):
        """
        Callback function for the image processed topic
        :param data: Image data in the ROS Image format
        """
        t1 = time.time()
        # Convert the image to the OpenCV format
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")

         # Update the header information
        header = Header()
        header.seq = data.header.seq
        header.stamp = data.header.stamp
        header.frame_id = data.header.frame_id
        # Update the header information in the message
        self.p.header = header

        # self.class_ids, __, self.boxes = self.detect(image, self.class_list, show=self.show)
        self.boxes, self.scores, self.class_ids = self.detector(image)
        if self.show:
            img = draw_detections(image, self.boxes, self.scores, self.class_ids)
            # cv2.rectangle(image, (100, 100), (200, 300), (255,0,0), 2)
            cv2.imshow('sign', img)
            cv2.waitKey(3)
        self.p.objects = self.class_ids
        self.p.num = len(self.class_ids)
        if self.p.num>=2:
            height1 = self.boxes[0][3]-self.boxes[0][1]
            width1 = self.boxes[0][2]-self.boxes[0][0]
            self.boxes[0][2] = width1
            self.boxes[0][3] = height1
            # print("height1, width1: ", height1, width1, self.class_names[self.class_ids[0]])
            height2 = self.boxes[1][3]-self.boxes[1][1]
            width2 = self.boxes[1][2]-self.boxes[1][0]
            self.boxes[1][2] = width2
            self.boxes[1][3] = height2
            self.p.box1 = self.boxes[0]
            self.p.box2 = self.boxes[1]
        elif self.p.num>=1:
            height1 = self.boxes[0][3]-self.boxes[0][1]
            width1 = self.boxes[0][2]-self.boxes[0][0]
            self.boxes[0][2] = width1
            self.boxes[0][3] = height1
            # print("height1, width1: ", height1, width1, self.class_names[self.class_ids[0]])
            self.p.box1 = self.boxes[0]

        # print(self.p)
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
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        
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
        # return input_tensor 


    def inference(self):
        start = time.perf_counter()
        # self.inputs[0].host = input_tensor
        trt_outputs = common.do_inference_v2(self.context,bindings=self.bindings,inputs = self.inputs,outputs=self.outputs,stream=self.stream)

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

