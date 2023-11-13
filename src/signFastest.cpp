#include "ros/ros.h"
#include "include/yolo-fastestv2.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "std_msgs/Header.h"
#include <chrono>
#include <vector>
#include <std_msgs/Float32MultiArray.h>
using namespace std::chrono;

class signFastest {
    public:
        signFastest(ros::NodeHandle& nh) 
            :it(nh)
        {
            nh.getParam("class_names", class_names);
            nh.getParam("confidence_thresholds", confidence_thresholds);
            nh.getParam("distance_thresholds", distance_thresholds);
            nh.getParam("/signFastest/showFlag", show);
            nh.getParam("/signFastest/printFlag", print);
            nh.getParam("/signFastest/printFlag", printDuration); //printDuration
            std::string model;
            nh.getParam("model", model);
            std::cout << "showFlag: " << show << std::endl;
            std::cout << "printFlag: " << print << std::endl;
            std::cout << "printDuration: " << printDuration << std::endl;
            std::cout << "class_names: " << class_names.size() << std::endl;
            std::cout << "confidence_thresholds: " << confidence_thresholds.size() << std::endl;
            std::cout << "distance_thresholds: " << distance_thresholds.size() << std::endl;
            std::cout << "model: " << model << std::endl;

            // std::string filePathParam = __FILE__;
            // size_t pos = filePathParam.rfind("/") + 1;
            // filePathParam.replace(pos, std::string::npos, "model/sissi753-opt.param");
            // const char* param = filePathParam.c_str();
            // std::string filePathBin = __FILE__;
            // pos = filePathBin.rfind("/") + 1;
            // filePathBin.replace(pos, std::string::npos, "model/sissi753-opt.bin");
            // const char* bin = filePathBin.c_str();
            // api.loadModel(param,bin);

            std::string filePathParam = __FILE__;
            size_t pos = filePathParam.rfind("/") + 1;
            filePathParam.replace(pos, std::string::npos, "model/" + model + ".param");
            const char* param = filePathParam.c_str();

            std::string filePathBin = __FILE__;
            pos = filePathBin.rfind("/") + 1;
            filePathBin.replace(pos, std::string::npos, "model/" + model + ".bin");
            const char* bin = filePathBin.c_str();

            api.loadModel(param, bin);

            pub = nh.advertise<std_msgs::Float32MultiArray>("sign", 10);
            std::cout <<"pub created" << std::endl;
            depth_sub = it.subscribe("/camera/depth/image_raw", 3, &signFastest::depthCallback, this);
            // wait for depth image
            ros::topic::waitForMessage<sensor_msgs::Image>("/camera/depth/image_raw", nh);
            sub = it.subscribe("/camera/image_raw", 3, &signFastest::imageCallback, this);
        }
        void depthCallback(const sensor_msgs::ImageConstPtr &msg) {
            try {
                cv_ptr_depth = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            } catch (cv_bridge::Exception &e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
        }
        void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
            if(printDuration) start = high_resolution_clock::now();
            try {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                // cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            } catch (cv_bridge::Exception &e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            api.detection(cv_ptr->image, boxes);
            std_msgs::Float32MultiArray sign_msg;
            sign_msg.layout.data_offset = 0;

            int hsy = 0;
            for (const auto &box : boxes) {
                int class_id = box.cate;
                float confidence = box.score;
                if (confidence >= confidence_thresholds[class_id]) {
                    double distance = computeMedianDepth(cv_ptr_depth->image, box)/1000; // in meters
                    if (distance <= distance_thresholds[class_id]) {
                        sign_msg.data.push_back(box.x1);
                        sign_msg.data.push_back(box.y1);
                        sign_msg.data.push_back(box.x2);
                        sign_msg.data.push_back(box.y2);
                        sign_msg.data.push_back(distance);
                        sign_msg.data.push_back(confidence);
                        sign_msg.data.push_back(static_cast<float>(class_id));
                        hsy++;
                    }
                }
            }
            if(hsy) {
                std_msgs::MultiArrayDimension dim;
                dim.label = "detections";
                dim.size = hsy;
                dim.stride = boxes.size() * 7;
                sign_msg.layout.dim.push_back(dim); 
            }
            // Publish Sign message
            pub.publish(sign_msg);
            if(printDuration) {
                stop = high_resolution_clock::now();
                duration = duration_cast<microseconds>(stop - start);
                ROS_INFO("sign durations: %ld", duration.count());
            }
            // for display
            if (show) {
                // Normalize depth img
                double maxVal;
                double minVal;
                cv::minMaxIdx(cv_ptr_depth->image, &minVal, &maxVal);
                cv_ptr_depth->image.convertTo(normalizedDepthImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
                for (int i = 0; i < boxes.size(); i++) {
                    char text[256];
                    int id = boxes[i].cate;
                    sprintf(text, "%s %.1f%%", class_names[id].c_str(), boxes[i].score * 100);
                    char text2[256];
                    double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i])/1000; 
                    sprintf(text2, "%s %.1fm", class_names[id].c_str(), distance);
                    int baseLine = 0;
                    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                    int x = boxes[i].x1;
                    int y = boxes[i].y1 - label_size.height - baseLine;
                    if (y < 0)
                        y = 0;
                    if (x + label_size.width > cv_ptr->image.cols)
                        x = cv_ptr->image.cols - label_size.width;

                    cv::rectangle(cv_ptr->image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                cv::Scalar(255, 255, 255), -1);
                    cv::putText(cv_ptr->image, text, cv::Point(x, y + label_size.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    cv::rectangle (cv_ptr->image, cv::Point(boxes[i].x1, boxes[i].y1), 
                                cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);

                    cv::rectangle(normalizedDepthImage, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                cv::Scalar(255, 255, 255), -1);
                    cv::putText(normalizedDepthImage, text2, cv::Point(x, y + label_size.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    cv::rectangle (normalizedDepthImage, cv::Point(boxes[i].x1, boxes[i].y1), 
                                cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
                }
                
                cv::imshow("normalized depth image", normalizedDepthImage);
                cv::imshow("image", cv_ptr->image);
                cv::waitKey(1);
            }
            if (print) {
                for (int i = 0; i < boxes.size(); i++) {
                    double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i])/1000;
                    std::cout<< "x1:" << boxes[i].x1<<", y1:"<<boxes[i].y1<<", x2:"<<boxes[i].x2<<", y2:"<<boxes[i].y2
                     <<", conf:"<<boxes[i].score<<", id:"<<boxes[i].cate<<", "<<class_names[boxes[i].cate]<<", dist:"<< distance <<std::endl;
                }
            }
        }
        
    private:
        yoloFastestv2 api;
        image_transport::Subscriber sub;
        image_transport::Subscriber depth_sub;
        image_transport::ImageTransport it;
        ros::Publisher pub;
        bool show;
        bool print;
        bool printDuration;
        cv_bridge::CvImagePtr cv_ptr;
        cv_bridge::CvImagePtr cv_ptr_depth;
        cv::Mat normalizedDepthImage;
        cv::Mat croppedDepth;

        std::vector<TargetBox> boxes;
        std::vector<TargetBox> boxes_depth;
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point stop;
        std::chrono::microseconds duration;

        std::vector<double> depths;
        std::vector<float> confidence_thresholds;
        std::vector<float> distance_thresholds;
        std::vector<std::string> class_names;
        double computeMedianDepth(const cv::Mat& depthImage, const TargetBox& box) {
            // Ensure the bounding box coordinates are valid
            int x1 = std::max(0, box.x1);
            int y1 = std::max(0, box.y1);
            int x2 = std::min(depthImage.cols, box.x2);
            int y2 = std::min(depthImage.rows, box.y2);
            croppedDepth = depthImage(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            std::vector<double> depths;
            for (int i = 0; i < croppedDepth.rows; ++i) {
                for (int j = 0; j < croppedDepth.cols; ++j) {
                    double depth = croppedDepth.at<float>(i, j);
                    if (depth > 100) {  // Only consider valid depth readings
                        depths.push_back(depth);
                    }
                }
            }
            if (depths.empty()) {
                return -1; 
            }
            std::sort(depths.begin(), depths.end());
            // double min = depths[0];
            // double max = depths[depths.size()-1];
            // std::cout << "min: " << min << " max: " << max << std::endl;

            //take the closest 20% of the pixels
            size_t index20Percent = depths.size() * 0.2;
            if (index20Percent <= 0) { // if there are less than 20% valid pixels
                return depths[0];
            }
            if (index20Percent % 2) { // if odd
                    return depths[index20Percent / 2];
                }
                return 0.5 * (depths[(index20Percent - 1) / 2] + depths[index20Percent / 2]);
            }
};

int main(int argc, char **argv) {
    int opt;
    
    // Initialize ROS node and publisher
    ros::init(argc, argv, "object_detector");
    ros::NodeHandle nh;
    
    signFastest signFastest(nh);
    //define rate
    ros::Rate loop_rate(25);

    // Spin ROS node
    while(ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}