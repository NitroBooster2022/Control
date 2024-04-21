#include "ros/ros.h"
#include "yolo-fastestv2.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "std_msgs/Header.h"
#include <chrono>
#include <vector>
#include <std_msgs/Float32MultiArray.h>
#include "utils/Lane.h"
#include <librealsense2/rs.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Imu.h>
#include <thread>
#include "LaneDetector.hpp"
#include "SignFastest.hpp"

using namespace std::chrono;

class CameraNode {
    public:
        CameraNode(ros::NodeHandle& nh) 
            :it(nh), Sign(nh), Lane(nh), align_to_color(RS2_STREAM_COLOR)
        {
            depthImage = cv::Mat::zeros(480, 640, CV_16UC1);
            colorImage = cv::Mat::zeros(480, 640, CV_8UC3);
            depth_frame = rs2::frame();
            color_frame = rs2::frame();
            realsense_imu_msg = sensor_msgs::Imu();
            data = rs2::frameset();

            // Declare RealSense pipeline, encapsulating the actual device and sensors
            cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
            cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
            cfg.enable_stream(RS2_STREAM_GYRO);
            cfg.enable_stream(RS2_STREAM_ACCEL);
            pipe.start(cfg);

            std::cout.precision(4);
            realsense_imu_pub = nh.advertise<sensor_msgs::Imu>("/camera/imu", 2);
            color_pub = nh.advertise<sensor_msgs::Image>("/camera/color/image_raw", 1);
            depth_pub = nh.advertise<sensor_msgs::Image>("/camera/depth/image_rect_raw", 1);
            std::cout <<"pub created" << std::endl; 
        }
        SignFastest Sign;
        LaneDetector Lane;

        rs2::pipeline pipe;
        rs2::config cfg;

        ros::Publisher realsense_imu_pub;
        ros::Publisher color_pub, depth_pub;
        sensor_msgs::ImagePtr color_msg, depth_msg;

        image_transport::Subscriber sub;
        image_transport::Subscriber depth_sub;
        image_transport::ImageTransport it;
        bool show;
        bool print;
        bool printDuration;
        bool hasDepthImage;
        cv_bridge::CvImagePtr cv_ptr;
        cv_bridge::CvImagePtr cv_ptr_depth;
        cv::Mat colorImage;
        cv::Mat depthImage;
        rs2::frame color_frame;
        rs2::frame depth_frame;
        sensor_msgs::Imu realsense_imu_msg;
        rs2::frameset data;
        rs2::frame gyro_frame;
        rs2::frame accel_frame;
        rs2::align align_to_color;
        void depthCallback(const sensor_msgs::ImageConstPtr &msg) {
            cv_ptr_depth = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        void laneCallback() {
        }
        void imageCallback() {
        }
        void get_frames() {
            while (true) {
                data = pipe.wait_for_frames();
                auto aligned_frames = align_to_color.process(data);
                color_frame = aligned_frames.get_color_frame();
                depth_frame = aligned_frames.get_depth_frame();
                gyro_frame = data.first_or_default(RS2_STREAM_GYRO);
                accel_frame = data.first_or_default(RS2_STREAM_ACCEL);
                if (!color_frame || !depth_frame) {
                    ROS_INFO("No frame received");
                    continue;
                }
                colorImage = cv::Mat(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
                depthImage = cv::Mat(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

                // Convert gyro and accel frames to ROS Imu message
                realsense_imu_msg.header.stamp = ros::Time::now();
                realsense_imu_msg.header.frame_id = "imu_link";
                float *gyro_data = (float *)gyro_frame.get_data();
                realsense_imu_msg.angular_velocity.x = gyro_data[0];
                realsense_imu_msg.angular_velocity.y = gyro_data[1];
                realsense_imu_msg.angular_velocity.z = gyro_data[2];

                float *accel_data = (float *)accel_frame.get_data();
                realsense_imu_msg.linear_acceleration.x = accel_data[0];
                realsense_imu_msg.linear_acceleration.y = accel_data[1];
                realsense_imu_msg.linear_acceleration.z = accel_data[2];
                realsense_imu_pub.publish(realsense_imu_msg);
                // ROS_INFO("frame received");
                // Convert color Mat to ROS Image message
                color_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", colorImage).toImageMsg();

                // Convert depth Mat to ROS Image message
                depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", depthImage).toImageMsg();
                cv_ptr_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);

                // Publish color, depth, and imu data
                color_pub.publish(color_msg);
                depth_pub.publish(depth_msg);
            }
        }
        void runLane() {
            static ros::Rate lane_rate(30);
            while(ros::ok()) {
                // ros::spinOnce();
                laneCallback();
                lane_rate.sleep();
            }
        }
};

int main(int argc, char **argv) {
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    int opt;

    // Initialize ROS node and publisher
    ros::init(argc, argv, "object_detector2");
    ros::NodeHandle nh;

    CameraNode CameraNode(nh);
    //define rate
    ros::Rate loop_rate(25);

    std::thread t1(&CameraNode::runLane, &CameraNode);
    std::thread t2(&CameraNode::get_frames, &CameraNode); // this already starts the thread i think
    // Spin ROS node
    while(ros::ok()) {
        // ros::spinOnce();
        CameraNode.imageCallback();
        // loop_rate.sleep();
    }
    t2.join();

    return 0;
}