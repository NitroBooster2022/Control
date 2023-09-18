#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <chrono>
// #include <raspicam/raspicam_cv.h>
// #include "/home/pi/Documents/Brain_ROS/devel/include/utils/Lane.h"
// #include "/home/pi/Documents/Brain_ROS/devel/include/utils/Sign.h"
#include "utils/Lane.h"
#include "utils/Sign.h"
#include "include/yolo-fastestv2.h"
#include <thread>
#include <mutex>
// using namespace std::chrono;

std::mutex mtx;
cv::Size imgSize;
bool stopline;
cv::Mat image;

int h = 480, w = 640;

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
cv::Mat cv_image;

static const char* class_names[] = {
        "oneway", "highwayentrance", "stopsign", "roundabout", "park", "crosswalk", "noentry", "highwayexit", "priority",
                "lights","block","pedestrian","car"
    };


std::vector<int> extract_lanes(cv::Mat hist_data) {
    std::vector<int> lane_indices;
    int previous_value = 0;
    for (int idx = 0; idx < hist_data.cols; ++idx) {
        int value = hist_data.at<int>(0, idx);
        if (value >= 1500 && previous_value == 0) {
            lane_indices.push_back(idx);
            previous_value = 255;
        } else if (value == 0 && previous_value == 255) {
            lane_indices.push_back(idx);
            previous_value = 0;
        }
    }
    if (lane_indices.size() % 2 == 1) {
        lane_indices.push_back(640 - 1);
    }
    return lane_indices;
}

double optimized_histogram(cv::Mat image, bool show = false) {
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

    std::vector<int> lanes = extract_lanes(hist);
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

    if (show) {
        cv::Mat padded_thresh = cv::Mat::zeros(480, 640, CV_8UC1);

        cv::Mat roi = padded_thresh(cv::Range(384, 384+thresh.rows), cv::Range::all());
        thresh.copyTo(roi);
        if (stopline) {
            cv::putText(padded_thresh, "Stopline detected!", cv::Point(static_cast<int>(w * 0.5), static_cast<int>(h * 0.5)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        cv::line(image, cv::Point(static_cast<int>(center), image.rows), cv::Point(static_cast<int>(center), static_cast<int>(0.8 * image.rows)), cv::Scalar(0, 0, 255), 5);
        cv::Mat add;
        cv::cvtColor(padded_thresh, add, cv::COLOR_GRAY2BGR);
        cv::imshow("Lane", image + add);
        cv::waitKey(1);
    }
    return center;
}
static void laneDetection(ros::Publisher *pub, double rate) {
    std::chrono::milliseconds sleep_duration(static_cast<int>(1000/rate));
    while (ros::ok()) {
        auto start = std::chrono::steady_clock::now();
        mtx.lock();
        cv::Mat local_cv_image = cv_image.clone();
        mtx.unlock();
        double center = optimized_histogram(local_cv_image);
        utils::Lane lane_msg;
        lane_msg.center = center;
        lane_msg.stopline = stopline;
        lane_msg.header.stamp = ros::Time::now();
        pub->publish(lane_msg);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        std::this_thread::sleep_for(sleep_duration-elapsed);
    }
}

static void signDetection(yoloFastestv2 *api, ros::Publisher *pub, double rate) {
    std::chrono::milliseconds sleep_duration(static_cast<int>(1000/rate));
    while (ros::ok()) {
        auto start = std::chrono::steady_clock::now();
        mtx.lock();
        cv::Mat local_cv_image = cv_image.clone();
        mtx.unlock();
        std::vector<TargetBox> boxes;
        api->detection(local_cv_image, boxes);
        utils::Sign sign_msg;
        sign_msg.header.stamp = ros::Time::now();
        sign_msg.header.frame_id = "camera_frame"; 

        sign_msg.num = boxes.size();

        int bb = 0;
        for (const auto &box : boxes) {
            sign_msg.objects.push_back(box.cate);
            if(bb==0){
                sign_msg.box1.push_back(box.x1);
                sign_msg.box1.push_back(box.y1);
                sign_msg.box1.push_back(box.x2-box.x1);
                sign_msg.box1.push_back(box.y2-box.y1);
            } else if (bb==1){
                sign_msg.box2.push_back(box.x1);
                sign_msg.box2.push_back(box.y1);
                sign_msg.box2.push_back(box.x2-box.x1);
                sign_msg.box2.push_back(box.y2-box.y1);
            } else if (bb == 2) {
                sign_msg.box3.push_back(box.x1);
                sign_msg.box3.push_back(box.y1);
                sign_msg.box3.push_back(box.x2-box.x1);
                sign_msg.box3.push_back(box.y2-box.y1);
            }
            sign_msg.confidence.push_back(box.score);
            bb++;
        }
        // Publish Sign message
        pub->publish(sign_msg);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        std::this_thread::sleep_for(sleep_duration-elapsed);
    }
}
// static void laneDetectionCallback(const ros::TimerEvent& event, ros::Publisher *pub) {
//     double center = optimized_histogram(cv_image);
//     utils::Lane lane_msg;
//     lane_msg.center = center;
//     lane_msg.stopline = stopline;
//     lane_msg.header.stamp = ros::Time::now();
//     pub->publish(lane_msg);
// }
// static void signDetectionCallback(const ros::TimerEvent& event, yoloFastestv2 *api, ros::Publisher *pub) {
//     std::vector<TargetBox> boxes;
//     api->detection(cv_image, boxes);

//     utils::Sign sign_msg;
//     sign_msg.header.stamp = ros::Time::now();
//     sign_msg.header.frame_id = "camera_frame"; 

//     sign_msg.num = boxes.size();

//     int bb = 0;
//     for (const auto &box : boxes) {
//         sign_msg.objects.push_back(box.cate);
//         if(bb==0){
//             sign_msg.box1.push_back(box.x1);
//             sign_msg.box1.push_back(box.y1);
//             sign_msg.box1.push_back(box.x2-box.x1);
//             sign_msg.box1.push_back(box.y2-box.y1);
//         } else if (bb==1){
//             sign_msg.box2.push_back(box.x1);
//             sign_msg.box2.push_back(box.y1);
//             sign_msg.box2.push_back(box.x2-box.x1);
//             sign_msg.box2.push_back(box.y2-box.y1);
//         } else if (bb == 2) {
//             sign_msg.box3.push_back(box.x1);
//             sign_msg.box3.push_back(box.y1);
//             sign_msg.box3.push_back(box.x2-box.x1);
//             sign_msg.box3.push_back(box.y2-box.y1);
//         }
//         sign_msg.confidence.push_back(box.score);
//         bb++;
//     }
//     // Publish Sign message
//     pub->publish(sign_msg);
// }

static void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_detector");
    yoloFastestv2 api;
    std::string filePathParam = __FILE__;
    size_t pos = filePathParam.rfind("/") + 1;
    filePathParam.replace(pos, std::string::npos, "model/amy357s-opt.param");
    const char* param = filePathParam.c_str();
    std::string filePathBin = __FILE__;
    pos = filePathBin.rfind("/") + 1;
    filePathBin.replace(pos, std::string::npos, "model/amy357s-opt.bin");
    const char* bin = filePathBin.c_str();
    api.loadModel(param,bin);
    ros::NodeHandle nh;
    ros::Publisher lane_pub = nh.advertise<utils::Lane>("lane", 3);
    ros::Publisher sign_pub = nh.advertise<utils::Sign>("sign", 3);
    
    cv_image = cv::Mat::zeros(480, 640, CV_8UC3);
    image = cv::Mat::zeros(480, 640, CV_8UC3);

    double lane_pub_rate = 25.0; // Adjust this value for lane_pub rate
    double sign_pub_rate = 1.0;
    // ros::Timer lane_timer = nh.createTimer(ros::Duration(1.0 / lane_pub_rate), [&](const ros::TimerEvent& event) { laneDetectionCallback(event, &lane_pub); });
    // ros::Timer sign_timer = nh.createTimer(ros::Duration(1.0 / sign_pub_rate), [&](const ros::TimerEvent& event) { signDetectionCallback(event, &api, &sign_pub); });
    std::thread lane_thread(laneDetection, &lane_pub, lane_pub_rate);
    std::thread sign_thread(signDetection, &api, &sign_pub, sign_pub_rate);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/automobile/image_raw", 1, imageCallback);
    
    stopline = false;

    while(ros::ok()) {
        ros::spinOnce();
    }
    lane_thread.join();
    sign_thread.join();
    return 0;
}