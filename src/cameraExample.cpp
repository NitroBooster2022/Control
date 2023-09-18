#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <chrono>
#include "utils/Lane.h"
#include "utils/Sign.h"
#include "include/yolo-fastestv2.h"
using namespace std::chrono;

class SignDetector {
public:
    SignDetector(ros::NodeHandle& nh)
        : it(nh) {
        sign_pub = nh.advertise<utils::Sign>("sign", 1000);
        // image_sub = it.subscribe("automobile/image_raw", 1, &SignDetector::imageCallback, this);
        class_names = { "oneway", "highwayentrance", "stopsign", "roundabout", "park", "crosswalk", 
                        "noentry", "highwayexit", "priority", "lights","block","pedestrian","car" };
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        auto start = high_resolution_clock::now();
        // Convert ROS image to OpenCV image
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Perform object detection
        std::vector<TargetBox> boxes;
        api.detection(cv_ptr->image, boxes);

        // Fill Sign message with all detected objects
        utils::Sign sign_msg;
        sign_msg.header.stamp = ros::Time::now();
        sign_msg.header.frame_id = "camera_frame"; // Set to appropriate frame_id if needed

        sign_msg.num = boxes.size();

        for (const auto& box : boxes) {
            sign_msg.objects.push_back(box.cate);
            sign_msg.box1.push_back(box.x1);
            sign_msg.box1.push_back(box.y1);
            sign_msg.box2.push_back(box.x2);
            sign_msg.box2.push_back(box.y2);
            sign_msg.confidence.push_back(box.score);
        }

        // Publish Sign message
        sign_pub.publish(sign_msg);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "sign durations: " << duration.count() << std::endl;
    }

private:
    yoloFastestv2 api;
    image_transport::ImageTransport it;
    // image_transport::Subscriber image_sub;
    ros::Publisher sign_pub;
    std::vector<std::string> class_names;
};

class LaneDetector {
public:
    LaneDetector(ros::NodeHandle& nh) : it(nh) {
        // image_sub = it.subscribe("/automobile/image_raw", 1, &LaneDetector::imageCallback, this);
        // image_pub = it.advertise("/automobile/image_modified", 1);
        lane_pub = nh.advertise<utils::Lane>("/lane", 1);
        image = cv::Mat::zeros(480, 640, CV_8UC1);
        stopline = false;
        dotted = false;
        pl = 320;
        auto start = high_resolution_clock::now();
        ros::Rate rate(15); 
        while (ros::ok()) {
            ros::spinOnce();
            rate.sleep();
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        auto start = high_resolution_clock::now();
        try {
            cv::Mat cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
            // auto start = high_resolution_clock::now();
            double center = optimized_histogram(cv_image);
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
            // sensor_msgs::ImagePtr modified_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image).toImageMsg();
            // image_pub.publish(modified_msg);
            utils::Lane lane_msg;
            lane_msg.center = center;
            lane_msg.stopline = stopline;
            lane_msg.header.stamp = ros::Time::now();
            lane_pub.publish(lane_msg);

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "lane durations: " << duration.count() << std::endl;
    }

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

    double optimized_histogram(cv::Mat image, bool show = true) {
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
            cv::imshow("Lane", image + add);
            cv::waitKey(1);
        }
        return center;
    }

private:
    // ros::NodeHandle nh;
    image_transport::ImageTransport it;
    // image_transport::Subscriber image_sub;
    image_transport::Publisher image_pub;
    ros::Publisher lane_pub;
    double num_iterations = 1;
    double total;
    cv::Mat maskh, masks, image, maskd;
    bool stopline, dotted;
    int pl;
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
    void addSquare(cv::Mat& image) {
        cv::Point top_left(100, 100);
        cv::Point bottom_right(200, 200);
        cv::Scalar color(0, 255, 0); // Green
        int thickness = 2;
        cv::rectangle(image, top_left, bottom_right, color, thickness);
    }
};

class CombinedDetector : public LaneDetector, public SignDetector {
public:
    CombinedDetector(ros::NodeHandle nh)
        : SignDetector(nh), LaneDetector(nh), it(nh) {
    }

    void laneDetectorCallback(const sensor_msgs::ImageConstPtr& msg) {
        LaneDetector::imageCallback(msg);
    }

    void signDetectorCallback(const sensor_msgs::ImageConstPtr& msg) {
        SignDetector::imageCallback(msg);
    }

private:
    image_transport::ImageTransport it;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "camera_example");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    
    CombinedDetector combined_detector(nh);
    
    // Use boost::bind to pass member function pointers as callbacks
    image_transport::Subscriber sub_lane = it.subscribe("/automobile/camera1/image_raw", 1, 
        boost::bind(&CombinedDetector::laneDetectorCallback, &combined_detector, _1));
    
    image_transport::Subscriber sub_sign = it.subscribe("/automobile/camera1/image_raw", 1, 
        boost::bind(&CombinedDetector::signDetectorCallback, &combined_detector, _1));
    
    ros::spin();
    
    return 0;
}