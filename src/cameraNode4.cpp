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

using namespace std::chrono;

class CameraNode {
    public:
        CameraNode(ros::NodeHandle& nh) 
            :it(nh), align_to_color(RS2_STREAM_COLOR)
        {
            normalizedDepthImage = cv::Mat::zeros(480, 640, CV_8UC1);
            depthImage = cv::Mat::zeros(480, 640, CV_16UC1);
            colorImage = cv::Mat::zeros(480, 640, CV_8UC3);
            // croppedDepth = cv::Mat::zeros(480, 640, CV_16UC1);
            depth_frame = rs2::frame();
            color_frame = rs2::frame();
            imu_msg = sensor_msgs::Imu();
            data = rs2::frameset();

            // Declare RealSense pipeline, encapsulating the actual device and sensors
            cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
            cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
            cfg.enable_stream(RS2_STREAM_GYRO);
            cfg.enable_stream(RS2_STREAM_ACCEL);
            pipe.start(cfg);

            std::cout.precision(4);
            imu_pub = nh.advertise<sensor_msgs::Imu>("/camera/imu", 2);
            color_pub = nh.advertise<sensor_msgs::Image>("/camera/color/image_raw", 1);
            depth_pub = nh.advertise<sensor_msgs::Image>("/camera/depth/image_rect_raw", 1);
            std::cout <<"pub created" << std::endl; 
        }

        rs2::pipeline pipe;
        rs2::config cfg;
        void depthCallback(const sensor_msgs::ImageConstPtr &msg) {
            cv_ptr_depth = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        void laneCallback() {
            // double center = optimized_histogram(cv_ptr->image, show, print);
            double center = optimized_histogram(colorImage, show, print);
            utils::Lane lane_msg;
            lane_msg.center = center;
            lane_msg.stopline = stopline;
            lane_msg.header.stamp = ros::Time::now();
            lane_pub.publish(lane_msg);
            // Publish Sign message
        }
        void imageCallback() {
            // std::cout << "cb" << std::endl;
            if(printDuration) start = high_resolution_clock::now();
            // cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            // if (!cv_ptr) {
            //     ROS_ERROR("cv_bridge failed to convert image");
            //     return;
            // }
            // api.detection(cv_ptr->image, boxes);
            api.detection(colorImage, boxes);
            std_msgs::Float32MultiArray sign_msg;
            sign_msg.layout.data_offset = 0;

            int hsy = 0;
            for (const auto &box : boxes) {
                int class_id = box.cate;
                float confidence = box.score;
                if (confidence >= confidence_thresholds[class_id]) {
                    double distance;
                    if(hasDepthImage) {
                        // distance = computeMedianDepth(cv_ptr_depth->image, box)/1000; // in meters
                        // ROS_INFO("depthImage size: %d, %d", depthImage.rows, depthImage.cols);
                        // double min, max;
                        // //print min and max values in depth image
                        // cv::minMaxIdx(depthImage, &min, &max);
                        // std::cout << "minVal: " << min << ", maxVal: " << max << std::endl;
                        distance = computeMedianDepth(depthImage, box)/1000; // in meters
                    } else {
                        distance = -1;
                    }
                    if (!distance_makes_sense(distance, class_id, box.x1, box.y1, box.x2, box.y2)) continue;
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
            if(hsy) {
                std_msgs::MultiArrayDimension dim;
                dim.label = "detections";
                dim.size = hsy;
                dim.stride = boxes.size() * 7;
                sign_msg.layout.dim.push_back(dim); 
            }
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
                if (hasDepthImage) {
                    // cv::minMaxIdx(cv_ptr_depth->image, &minVal, &maxVal);
                    cv::minMaxIdx(depthImage, &minVal, &maxVal);
                    // cv_ptr_depth->image.convertTo(normalizedDepthImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
                    depthImage.convertTo(normalizedDepthImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
                }
                for (int i = 0; i < boxes.size(); i++) {
                    char text[256];
                    int id = boxes[i].cate;
                    sprintf(text, "%s %.1f%%", class_names[id].c_str(), boxes[i].score * 100);
                    char text2[256];
                    if (hasDepthImage) {
                        // double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i])/1000; 
                        double distance = computeMedianDepth(depthImage, boxes[i])/1000;
                       sprintf(text2, "%s %.1fm", class_names[id].c_str(), distance);
                    }
                    int baseLine = 0;
                    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                    int x = boxes[i].x1;
                    int y = boxes[i].y1 - label_size.height - baseLine;
                    if (y < 0)
                        y = 0;
                    // if (x + label_size.width > cv_ptr->image.cols)
                        // x = cv_ptr->image.cols - label_size.width;
                    if (x + label_size.width > colorImage.cols)
                        x = colorImage.cols - label_size.width;

                    // cv::rectangle(cv_ptr->image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    //             cv::Scalar(255, 255, 255), -1);
                    // cv::putText(cv_ptr->image, text, cv::Point(x, y + label_size.height),
                    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    // cv::rectangle (cv_ptr->image, cv::Point(boxes[i].x1, boxes[i].y1), 
                    //             cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
                    cv::rectangle(colorImage, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                cv::Scalar(255, 255, 255), -1);
                    cv::putText(colorImage, text, cv::Point(x, y + label_size.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    cv::rectangle (colorImage, cv::Point(boxes[i].x1, boxes[i].y1),
                                cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
                    if(hasDepthImage) {
                        cv::rectangle(normalizedDepthImage, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                                cv::Scalar(255, 255, 255), -1);
                        cv::putText(normalizedDepthImage, text2, cv::Point(x, y + label_size.height),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                        cv::rectangle (normalizedDepthImage, cv::Point(boxes[i].x1, boxes[i].y1), 
                                cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
                    }
                }
                if(hasDepthImage) {
                    cv::imshow("normalized depth image", normalizedDepthImage);
                }
                // cv::imshow("sign_image", cv_ptr->image);
                cv::imshow("sign_image", colorImage);
                cv::waitKey(1);
            }
            if (print) {
                for (int i = 0; i < boxes.size(); i++) {
                    // double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i])/1000;
                    double distance = computeMedianDepth(depthImage, boxes[i])/1000;
                    std::cout<< "x1:" << boxes[i].x1<<", y1:"<<boxes[i].y1<<", x2:"<<boxes[i].x2<<", y2:"<<boxes[i].y2
                     <<", conf:"<<boxes[i].score<<", id:"<<boxes[i].cate<<", "<<class_names[boxes[i].cate]<<", dist:"<< distance <<", w:"<<boxes[i].x2-boxes[i].x1<<", h:"<<boxes[i].y2-boxes[i].y1<<std::endl;
                }
            }
        }

    // private:
        //lane 
        ros::Publisher imu_pub;
        ros::Publisher lane_pub;
        ros::Publisher color_pub, depth_pub;
        sensor_msgs::ImagePtr color_msg, depth_msg;
        double num_iterations = 1;
        double previous_center;
        double total;
        cv::Mat maskh, masks, image, maskd;
        bool stopline, dotted;
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

        yoloFastestv2 api;
        image_transport::Subscriber sub;
        image_transport::Subscriber depth_sub;
        image_transport::ImageTransport it;
        ros::Publisher pub;
        bool show;
        bool print;
        bool printDuration;
        bool hasDepthImage;
        cv_bridge::CvImagePtr cv_ptr;
        cv_bridge::CvImagePtr cv_ptr_depth;
        cv::Mat normalizedDepthImage;
        cv::Mat croppedDepth;
        cv::Mat colorImage;
        cv::Mat depthImage;
        rs2::frame color_frame;
        rs2::frame depth_frame;
        sensor_msgs::Imu imu_msg;
        rs2::frameset data;
        rs2::frame gyro_frame;
        rs2::frame accel_frame;
        rs2::align align_to_color;

        std::vector<TargetBox> boxes;
        std::vector<TargetBox> boxes_depth;
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point stop;
        std::chrono::microseconds duration;

        std::vector<double> depths;
        std::vector<float> confidence_thresholds;
        std::vector<float> distance_thresholds;
        std::vector<std::string> class_names;

        static constexpr double SIGN_H2D_RATIO = 31.57;
        static constexpr double LIGHT_W2D_RATIO = 41.87;
        static constexpr double CAR_H2D_RATIO = 90.15;

        double computeMedianDepth(const cv::Mat& iDepthImage, const TargetBox& box) {
            // std::cout << "size: " << iDepthImage.size() << std::endl;
            // double minVal, maxVal;
            // std::cout << "minVal0: " << minVal << ", maxVal0: " << maxVal << std::endl;
            // cv::minMaxIdx(iDepthImage, &minVal, &maxVal);
            // iDepthImage.convertTo(normalizedDepthImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            // cv::imshow("depth", normalizedDepthImage);
            // cv::waitKey(1);
            // Ensure the bounding box coordinates are valid
            int x1 = std::max(0, box.x1);
            int y1 = std::max(0, box.y1);
            int x2 = std::min(iDepthImage.cols, box.x2);
            int y2 = std::min(iDepthImage.rows, box.y2);
            // std::cout << "x1: " << x1 << ", y1: " << y1 << ", x2: " << x2 << ", y2: " << y2 << std::endl;
            croppedDepth = iDepthImage(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            // cv::minMaxLoc(croppedDepth, &minVal, &maxVal);
            // std::cout << "minVal1: " << minVal << ", maxVal1: " << maxVal << std::endl;
            // cv::Mat normalizedDepthImage2;
            // croppedDepth.convertTo(normalizedDepthImage2, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            // cv::imshow("depth", normalizedDepthImage2);
            // cv::waitKey(1);
            // std::cout << "croppedDepth size: " << croppedDepth.size() << "," << croppedDepth.rows << "," << croppedDepth.cols << std::endl;
            // //print the entire cropped depth image
            // for (int i = 0; i < croppedDepth.rows; ++i) {
            //     for (int j = 0; j < croppedDepth.cols; ++j) {
            //         std::cout << normalizedDepthImage2.at<float>(i, j) << " ";
            //     }
            //     std::cout << std::endl;
            // }
            std::vector<double> depths;
            for (int i = 0; i < croppedDepth.rows; ++i) {
                for (int j = 0; j < croppedDepth.cols; ++j) {
                    uint16_t depth = croppedDepth.at<uint16_t>(i, j);
                    if (depth > 0.0001)
                        // std::cout << "depth: " << depth << std::endl;
                    if (depth > 100) {  // Only consider valid depth readings
                        depths.push_back(static_cast<double>(depth));
                    }
                }
            }
            if (depths.empty()) {
                std::cout << "No valid depth readings found in the bounding box" << std::endl;
                return -1; 
            }
            // Find the median using std::nth_element
            size_t index20Percent = depths.size() * 0.2;
            std::nth_element(depths.begin(), depths.begin() + index20Percent, depths.end());
            if (index20Percent % 2) { // if odd
                // std::cout << "median: " << depths[index20Percent / 2] << std::endl;
                return depths[index20Percent / 2];
            }
            // std::cout << "median: " << 0.5 * (depths[(index20Percent - 1) / 2] + depths[index20Percent / 2]) << std::endl;
            return 0.5 * (depths[(index20Percent - 1) / 2] + depths[index20Percent / 2]);
        }

        std::vector<int> extract_lanes(cv::Mat &hist_data) {
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
        double optimized_histogram(cv::Mat &image, bool show = false, bool print = false) {
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
                cv::imshow("Lane", image + add);
                cv::waitKey(1);
            }
            if (print) {
                std::cout << "center: " << center << std::endl;
                std::cout << "thresh: " << threshold_value << std::endl;
            }
            previous_center = center;
            return center;
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
                imu_msg.header.stamp = ros::Time::now();
                imu_msg.header.frame_id = "imu_link";
                float *gyro_data = (float *)gyro_frame.get_data();
                imu_msg.angular_velocity.x = gyro_data[0];
                imu_msg.angular_velocity.y = gyro_data[1];
                imu_msg.angular_velocity.z = gyro_data[2];

                float *accel_data = (float *)accel_frame.get_data();
                imu_msg.linear_acceleration.x = accel_data[0];
                imu_msg.linear_acceleration.y = accel_data[1];
                imu_msg.linear_acceleration.z = accel_data[2];
                imu_pub.publish(imu_msg);
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


// dist:0.952, w:39, h:42
// dist:0.499, w:66, h:74
// dist:0.344, w:84, h:103
// dist:1.121, w:29, h:38

//stopsign
// dist:1.037, w:32, h:31
// dist:0.381, w:93, h:82
// dist:0.366, w:83, h:85
// dist:0.435, w:66, h:69
// dist:0.549, w:52, h:53
// dist:0.689, w:49, h:46
// dist:0.844, w:41, h:45
// dist:1.008, w:35, h:40
// dist:1.143, w:28, h:30

//light
// dist:0.401, w:89, h:188
// dist:0.459, w:79, h:201
// dist:0.478, w:72, h:180
// dist:0.562, w:81, h:213
// dist:0.616, w:74, h:184
// dist:0.675, w:73, h:176
// dist:0.755, w:60, h:161
// dist:0.865, w:73, h:147
// dist:0.9725, w:52, h:125
// dist:1.08, w:44, h:107
// dist:1.219, w:44, h:99
// dist:1.455, w:34, h:76

// car
// dist:0.573, w:362, h:149
// dist:0.644, w:341, h:151
// dist:0.704, w:344, h:134
// dist:0.784, w:317, h:112
// dist:0.873, w:284, h:100
// dist:0.94, w:266, h:96
// dist:1.067, w:239, h:89
// dist:1.197, w:212, h:77
// dist:1.256, w:202, h:69
// dist:1.334, w:189, h:67
// dist:1.435, w:167, h:55
// dist:1.531, w:166, h:62
// dist:1.627, w:155, h:52
// dist:1.754, w:133, h:52
// dist:1.933, w:143, h:44