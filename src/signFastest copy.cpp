#include "ros/ros.h"
#include "include/yolo-fastestv2.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "std_msgs/Header.h"
#include "utils/Sign.h"
#include <chrono>
using namespace std::chrono;

class signFastest {
    public:
        signFastest(ros::NodeHandle& nh, bool show_, bool print_, bool printDuration_) 
            :it(nh),
            show(show_),
            print(print_),
            printDuration(printDuration_)
        {
            std::string filePathParam = __FILE__;
            size_t pos = filePathParam.rfind("/") + 1;
            filePathParam.replace(pos, std::string::npos, "model/sissi753-opt.param");
            const char* param = filePathParam.c_str();

            std::string filePathBin = __FILE__;
            pos = filePathBin.rfind("/") + 1;
            filePathBin.replace(pos, std::string::npos, "model/sissi753-opt.bin");
            const char* bin = filePathBin.c_str();
            api.loadModel(param,bin);

            pub = nh.advertise<utils::Sign>("sign", 10);
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

            utils::Sign sign_msg;
            sign_msg.header.frame_id = "camera_frame"; 
            sign_msg.header.stamp = ros::Time::now();
            sign_msg.num = boxes.size();

            int hsy = 0;
            for (const auto &box : boxes) {
                double distance = computeMedianDepth(cv_ptr_depth->image, box)/1000; // dist from cam to front of car
                sign_msg.distances.push_back(distance);
                sign_msg.objects.push_back(box.cate);
                std::vector<std::vector<float>*> box_data = {&sign_msg.box1, &sign_msg.box2, &sign_msg.box3, &sign_msg.box4};
                if (hsy < 4) {
                    box_data[hsy]->push_back(box.x1);
                    box_data[hsy]->push_back(box.y1);
                    box_data[hsy]->push_back(box.x2 - box.x1);
                    box_data[hsy]->push_back(box.y2 - box.y1);
                }
                sign_msg.confidence.push_back(box.score);
                hsy++;
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
                    sprintf(text, "%s %.1f%%", class_names[boxes[i].cate], boxes[i].score * 100);
                    char text2[256];
                    sprintf(text2, "%s %.1fm", class_names[boxes[i].cate], sign_msg.distances[i]);
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
                    std::cout<<boxes[i].x1<<" "<<boxes[i].y1<<" "<<boxes[i].x2-boxes[i].x1<<" "<<boxes[i].y2-boxes[i].y1
                     <<" "<<boxes[i].score<<" "<<boxes[i].cate<<" "<<class_names[boxes[i].cate]<<" dist: "<<sign_msg.distances[i]<<std::endl;
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
        static const char* class_names[13];
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
        // cv::Mat croppedDepth;
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


const char* signFastest::class_names[13] = {
    "oneway", "highwayentrance", "stopsign", "roundabout", "park", "crosswalk",
    "noentry", "highwayexit", "priority", "lights", "block", "pedestrian", "car"
};
int main(int argc, char **argv) {
    int opt;
    bool showFlag = false;
    bool printFlag = false;
    bool printDuration = false;
    std::string modelnum = "11";
    
    // Loop thru command line args
    while ((opt = getopt(argc, argv, "hspd:m:")) != -1) {
        switch (opt) {
            case 's': 
                showFlag = true;
                break;
            case 'p': 
                printFlag = true;
                printDuration = true;
                break;
            case 'd': 
                printDuration = true;
                break;
            case 'm': 
                modelnum = optarg; 
                break;
            case 'h': 
                std::cout << "-s to display image\n";
                std::cout << "-p to print detection\n";
                std::cout << "-d to print duration\n";  // added this for clarity
                std::cout << "-m [model_number] to set the model number\n";
                exit(0);
            default:
                std::cerr << "Invalid argument\n";
                exit(1);
        }
    }

    // Initialize ROS node and publisher
    ros::init(argc, argv, "object_detector");
    ros::NodeHandle nh;
    signFastest signFastest(nh, showFlag, printFlag, printDuration);
    //define rate
    ros::Rate loop_rate(25);

    // Spin ROS node
    while(ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}