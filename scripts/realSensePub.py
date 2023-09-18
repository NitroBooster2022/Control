#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
from utils.msg import ImgInfo
import rospy
class RealSenseWrapper:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
    def getColor(self):
        try:
            print("start get color")
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            return color_image.astype(np.uint32)
        except Exception as e:
            print(e)
            pass
    
    def getDepth(self):
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            return depth_image.astype(np.uint32)
        except Exception as e:
            print(e)
            pass
    def Stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    rospy.init_node("publishRealSenseInfo")
    pub = rospy.Publisher("RealSenseInfo", ImgInfo,queue_size=10)

    imgInfo = ImgInfo()
    realSenseWrap = RealSenseWrapper()
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            imgInfo.colorImg = realSenseWrap.getColor().flatten()
            imgInfo.depImg = realSenseWrap.getDepth().flatten()
            pub.publish(imgInfo)
            rate.sleep()
            rospy.loginfo("colorImg: %s, depthImg: %s, type: %s",str(imgInfo.colorImg.shape),str(imgInfo.depImg.shape),imgInfo.colorImg.dtype)
    finally:
        realSenseWrap.Stop()