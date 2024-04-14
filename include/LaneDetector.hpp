#pragma once

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <chrono>
#include "utils/Lane.h"
#include <mutex>

using namespace std::chrono;

class LaneDetector {
public:
    LaneDetector(ros::NodeHandle& nh) : nh(nh), showflag(false), printflag(false), previous_center(320)
    {
        // image_sub = it.subscribe("camera/color/image_raw", 1, &LaneDetector::imageCallback, this);
        // image_pub = it.advertise("/automobile/image_modified", 1);
        lane_pub = nh.advertise<utils::Lane>("/lane", 1);
        image = cv::Mat::zeros(480, 640, CV_8UC1);
        stopline = false;
        dotted = false;
        std::string nodeName = ros::this_node::getName();
        nh.getParam(nodeName+"/showFlag", showflag);
        nh.getParam(nodeName+"/printFlag", printflag);
        nh.getParam(nodeName+"/pub", publish);
        // ros::Rate rate(40); 
        // while (ros::ok()) {
        //     ros::spinOnce();
        //     rate.sleep();
        // }
    }

    // private:
    ros::NodeHandle nh;
    // image_transport::ImageTransport it;
    // image_transport::Subscriber image_sub;
    // image_transport::Publisher image_pub;
    ros::Publisher lane_pub;
    utils::Lane lane_msg;
    double num_iterations = 1;
    double previous_center;
    double total;
    cv::Mat maskh, masks, image, maskd;
    bool stopline, dotted, publish;
    int h = 480, w = 640;
    std::vector<int> lanes;
    
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::Mat img_gray;
    cv::Mat img_roi;
    cv::Mat thresh;
    cv::Mat hist;
    cv::Mat img_rois;
    double threshold_value_stop;
    cv::Mat threshs;
    cv::Mat hists;
    bool showflag, printflag;
    std::mutex mutex;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv::Mat cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
        if (cv_image.empty()) {
            ROS_INFO("Empty image");
            return;
        }
        // auto start = high_resolution_clock::now();
        double center = optimized_histogram(cv_image, showflag, printflag);
        // auto stop = high_resolution_clock::now();
        // auto duration = duration_cast<microseconds>(stop - start);
        // total+=static_cast<double>(duration.count());
        // double avg_duration = total / num_iterations;
        // num_iterations++;

        // std::cout << "durations: " << duration.count() << std::endl;
        // std::cout << "avg: " << avg_duration << std::endl;
        // std::cout << "center: " << center << std::endl;

        // cv::imshow("Frame preview", cv_image);
        // cv::waitKey(1);

        // Publish the modified image
        sensor_msgs::ImagePtr modified_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image).toImageMsg();
        // image_pub.publish(modified_msg);
        utils::Lane lane_msg;
        lane_msg.center = center;
        lane_msg.stopline = stopline;
        lane_msg.header.stamp = ros::Time::now();
        if(publish) lane_pub.publish(lane_msg);
    }

    void publish_lane(const cv::Mat& image) {
        if(image.empty()) {
            ROS_WARN("empty image received in lane detector");
            return;
        }
        double center = optimized_histogram(image, showflag, printflag);
        lane_msg.center = center;
        lane_msg.stopline = stopline;
        lane_msg.header.stamp = ros::Time::now();
        lane_pub.publish(lane_msg);
    }

    // std::vector<int> extract_lanes(const cv::Mat& hist_data) {
    void extract_lanes(const cv::Mat& hist_data) {
        // std::vector<int> lane_indices;
        lanes.clear();
        int previous_value = 0;
        for (int idx = 0; idx < hist_data.cols; ++idx) {
            int value = hist_data.at<int>(0, idx);
            if (value >= 1500 && previous_value == 0) {
                lanes.push_back(idx);
                previous_value = 255;
            } else if (value == 0 && previous_value == 255) {
                lanes.push_back(idx);
                previous_value = 0;
            }
        }
        if (lanes.size() % 2 == 1) {
            lanes.push_back(640 - 1);
        }
        // return lane_indices;
    }

    double optimized_histogram(const cv::Mat& image, bool show = false, bool print = false) {
        stopline = false;
        cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

        // apply maskh
        img_roi = img_gray(cv::Rect(0, 384, 640, 96));
        // cv::imshow("L", img_roi);
        // cv::waitKey(1);
        cv::minMaxLoc(img_roi, &minVal, &maxVal, &minLoc, &maxLoc);
        double threshold_value = std::min(std::max(maxVal - 55.0, 30.0), 200.0);
        cv::threshold(img_roi, thresh, threshold_value, 255, cv::THRESH_BINARY);
        hist = cv::Mat::zeros(1, w, CV_32SC1);
        cv::reduce(thresh, hist, 0, cv::REDUCE_SUM, CV_32S);

        // apply masks
        // img_rois = img_gray(cv::Range(300, 340), cv::Range::all());
        // // cv::imshow("S", img_rois);
        // // cv::waitKey(1);
        // cv::minMaxLoc(img_roi, &minVal, &maxVal, &minLoc, &maxLoc); // Use img_roi or img_rois depending on your requirements
        // threshold_value_stop = std::min(std::max(maxVal - 65.0, 30.0), 200.0);
        
        // cv::threshold(img_rois, threshs, threshold_value_stop, 255, cv::THRESH_BINARY);
        // hists = cv::Mat::zeros(1, w, CV_32SC1);
        // cv::reduce(threshs, hists, 0, cv::REDUCE_SUM, CV_32S);

        // std::vector<int> stop_lanes = extract_lanes(hists);
        // for (size_t i = 0; i < stop_lanes.size() / 2; ++i) {
        //     if (abs(stop_lanes[2 * i] - stop_lanes[2 * i + 1]) > 370 && threshold_value > 30) {
        //         stopline = true;
        //         if (!show) return w / 2.0;
        //     }
        // }

        // std::vector<int> lanes = extract_lanes(hist);
        extract_lanes(hist);
        std::vector<double> centers;
        for (size_t i = 0; i < lanes.size() / 2; ++i) {
            if (abs(lanes[2 * i] - lanes[2 * i + 1])>350 && threshold_value>50){
                stopline = true;
                if (!show) return w / 2.0;
            }
            if (3 < abs(lanes[2 * i] - lanes[2 * i + 1])) {
                centers.push_back((lanes[2 * i] + lanes[2 * i + 1]) / 2.0);
            }
        }
        double center;
        if (centers.empty()) {
            center = w / 2.0;
        } else if (centers.size() == 1) {
            center = (centers[0] > (w / 2.0)) ? (centers[0] - 0) / 2 : (centers[0] * 2 + w) / 2;
        } else if (abs(centers[0] - centers.back()) < 200) {
            center = ((centers[0] + centers.back()) > w) ? ((centers[0] + centers.back()) / 2 + 0) / 2.0 : ((centers[0] + centers.back()) + w) / 2;
        } else {
            center = (centers[0] + centers.back()) / 2;
        }

        // if(std::abs(center - previous_center) > 250) {
        //     center = previous_center;
        // }
        // if (std::abs(center - 320) < 1) {
        //     double temp = center;
        //     center = previous_center;
        //     previous_center = temp;
        // } else {
        //     previous_center = center;
        // }

        if (show) {
            // Create the new cv::Mat object and initialize it with zeros
            cv::Mat padded_thresh = cv::Mat::zeros(480, 640, CV_8UC1);

            // Copy the truncated array into the new cv::Mat object
            cv::Mat roi = padded_thresh(cv::Range(384, 384+thresh.rows), cv::Range::all());
            thresh.copyTo(roi);
            if (stopline) {
                cv::putText(padded_thresh, "Stopline detected!", cv::Point(static_cast<int>(w * 0.5), static_cast<int>(h * 0.5)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
            if (dotted) {
                cv::putText(image, "DottedLine!", cv::Point(static_cast<int>(w*0.5), static_cast<int>(h * 0.5)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            cv::line(image, cv::Point(static_cast<int>(center), image.rows), cv::Point(static_cast<int>(center), static_cast<int>(0.8 * image.rows)), cv::Scalar(0, 0, 255), 5);
            cv::Mat add;
            cv::cvtColor(padded_thresh, add, cv::COLOR_GRAY2BGR);
            // cv::imshow("Lane", image + add);
            // cv::waitKey(1);
        }
        if (print) {
            std::cout << "center: " << center << std::endl;
            std::cout << "thresh: " << threshold_value << std::endl;
        }
        previous_center = center;
        return center;
    }

};