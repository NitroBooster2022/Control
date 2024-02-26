#include "ros/ros.h"
#include "include/yolo-fastestv2.h"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "cv_bridge/cv_bridge.h"
#include "std_msgs/Header.h"
#include "include/engine.h"
#include <chrono>
#include <vector>
#include <std_msgs/Float32MultiArray.h>

using namespace nvinfer1;
Options options;
Engine model(options);

class signTRT{
    public:
        
        // Specify what precision to use for inference
        // FP16 is approximately twice as fast as FP32.
        
        signTRT(ros::NodeHandle& nh):it(nh)//, inputDims(inputDims)
        {
            
            //loading from yaml file
            nh.getParam("class_names",class_names);
            nh.getParam("confidence_thresholds", confidence_thresholds);
            nh.getParam("distance_thresholds", distance_thresholds);
            nh.getParam("/signFastest/showFlag",show);
            nh.getParam("/signFastest/printFlag",print);
            nh.getParam("signFastest/printDuration",printDuration);
            std::string model_name = "best";
            // nh.getParam("model",model_name);
            std::cout << "showFlag: " << show << std::endl;
            std::cout << "printFlag: " << print << std::endl;
            std::cout << "printDuration: " << printDuration << std::endl;
            std::cout << "class_names: " << class_names.size() << std::endl;
            std::cout << "confidence_thresholds: " << confidence_thresholds.size() << std::endl;
            std::cout << "distance_thresholds: " << distance_thresholds.size() << std::endl;
            std::cout << "model: " << model_name << std::endl;
            std::cout << "ck1"<< std::endl;
            //get the model path
            std::string filePathParam = __FILE__;
            size_t pos = filePathParam.rfind("/") + 1;
            filePathParam.replace(pos,std::string::npos, "model/" + model_name + ".engine");
            // const char* modelPath = filePathParam.c_str();
            //input image dimension
            inputDims = {640,640};
            //initialize engine
            

            // engine(options); //create engine object
            model.m_engineName = filePathParam;
            // Load the TensorRT engine file from disk
            succ = model.loadNetwork();
            if (!succ) {
                throw std::runtime_error("Unable to load TRT engine.");
            }

            //topics management
            pub = nh.advertise<std_msgs::Float32MultiArray>("sign",10);
            std::cout << "pub created" << std::endl;
            
            // depth_sub = it.subscribe("/camera/depth/image_raw",3,&signTRT::depthCallback,this);
            // ros::topic::waitForMessage<sensor_msgs::Image>("/camera/depth/image_raw",nh);
            sub = it.subscribe("/camera/color/image_raw",3,&signTRT::imageCallback,this);

        }
        void depthCallback(const sensor_msgs::ImageConstPtr& msg){
            std::cout << "dpt callback"<< std::endl;
            try{
                cv_ptr_depth = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_32FC1);
            }
            catch(cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s",e.what());
                return;
            }
        }
        void imageCallback(const sensor_msgs::ImageConstPtr& msg){
            std::cout<<"img callback"<< std::endl;
            if(printDuration) start = std::chrono::high_resolution_clock::now();
            std::cout << "ck7"<< std::endl;
            // try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            // }
            // catch (cv_bridge::Exception& e){
            //     ROS_ERROR("cv_bridge exception: %s",e.what());
            //     return;
            // }
            //input image
            std::cout << "ck1" << std::endl;
            std::cout << "ck1" << std::endl;
            img.upload(cv_ptr->image);
            // std::cout<<cv_ptr->image<<std::endl;
            //detection params
            // inputDims = model.getInputDims();
            batchSize = options.optBatchSize;
            std::cout << "ck8"<< std::endl;
            // resized = Engine::resizeKeepAspectRatioPadRightBottom(img,640,480); //inputDims[0].d[1], inputDims[0].d[2])
            std::cout << "ck8"<< std::endl;
            for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination of the two in order to maintain the aspect ratio
            // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
                resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDims[0],inputDims[1]);
            // You could also perform a resize operation without maintaining aspect ratio with the use of padding by using the following instead:
//            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
                input.emplace_back(std::move(resized));
            }
            inputs.emplace_back(std::move(input));
            
            std::cout << "ck0"<< std::endl;


            succ = model.runInference(inputs,featureVectors);

            //convert featureVectors into output boxes
            std::vector<cv::Rect> tmp_boxes;
            std::vector<float> scores;
            std::vector<int> classes;
            std::cout << featureVectors[0].size()<< std::endl;
            int x1, y1, x2, y2;
            outputDims = {model.getOutputDims()[0].d[0],model.getOutputDims()[0].d[1],model.getOutputDims()[0].d[2],model.getOutputDims()[0].d[3]};
            std::cout << outputDims[0] <<std::endl;
            std::cout << outputDims[1] <<std::endl;
            std::cout << outputDims[2] <<std::endl;
            std::cout << outputDims[3] <<std::endl;
            std::cout << featureVectors[0][0].size()<< std::endl;
            for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
                for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
                    std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
                    int i = 0;
                    for (const auto &e:  featureVectors[batch][outputNum]) {
                        std::cout << e << " ";
                        if (++i == 100) {
                            std::cout << "...";
                            break;
                        }
                    }
                    std::cout << "\n" << std::endl;
                }
            }


            for (size_t outputNum = 0; outputNum < featureVectors[0].size(); ++outputNum) {
                double minVal, maxVal = 0.0;
                cv::Point minLoc; cv::Point maxLoc;
                cv::Mat rowWithRange = cv::Mat();//(cv::Rect(range.start, 0, range.size(), 1));
                std::cout << "ck2"<< std::endl;
                cv::minMaxLoc(rowWithRange,&minVal, &maxVal, &minLoc, &maxLoc);
                std::cout << "ck2"<< std::endl;
                cv::Point2i intPoint(maxLoc.x, maxLoc.y);
                std::cout << "ck2"<< std::endl;
                if (maxVal >= confidence_thresholds[intPoint.x]) {
                    scores.push_back(maxVal);
                    classes.push_back(intPoint.x);
                    std::cout << "ck3"<< std::endl;
                    x1 = featureVectors[0][outputNum][0]* img.cols;
                    y1 = featureVectors[0][outputNum][1] * img.rows;
                    y2 = y1 + featureVectors[0][outputNum][3] *  img.rows;
                    x2 = x1 + featureVectors[0][outputNum][2] *  img.cols;
                    std::cout << "ck4"<< std::endl;
                    tmp_boxes.push_back(cv::Rect(x1,y1,x2,y2));
                    std::cout << "ck5"<< std::endl;
                }
            }

            featureVectors.clear();
            
            //NMS suppression
            std::vector<int> indices;
            cv::dnn::NMSBoxes(tmp_boxes, scores, 0.25, 0.45, indices, 0.5);
            TargetBox tmp_box;
            for (int i = 0; i < indices.size();i++) {
            tmp_box.cate = classes[indices[i]];
            tmp_box.score = scores[indices[i]];
            tmp_box.x1 = tmp_boxes[indices[i]].x;
            tmp_box.y1 = tmp_boxes[indices[i]].y;
            tmp_box.x2 = tmp_boxes[indices[i]].x +  tmp_boxes[indices[i]].width; 
            tmp_box.y2 = tmp_boxes[indices[i]].y +  tmp_boxes[indices[i]].height;
            boxes.push_back(tmp_box);
            }

            int hsy = 0;
            //what to publish in topic
            for (const auto& box:boxes) {
                int class_id = box.cate;
                float confidence = box.score;
                if (confidence >= confidence_thresholds[class_id]){
                    double distance = computeMedianDepth(cv_ptr_depth->image,box)/1000;
                    if (distance <= distance_thresholds[class_id]){
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

            if(hsy){
                std_msgs::MultiArrayDimension dim;
                dim.label = "detections";
                dim.size = hsy;
                dim.stride = boxes.size() *7;
                sign_msg.layout.dim.push_back(dim);
            }
            //publish message
            pub.publish(sign_msg);
            if (printDuration) {
                stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
                ROS_INFO("sign durations: %ld", duration.count());
            }

            //display
            if(show) {
                double maxVal;
                double minVal;
                cv::minMaxIdx(cv_ptr_depth->image,&minVal,&maxVal);
                cv_ptr_depth->image.convertTo(normalizedDepthImage,CV_8U,255.0/(maxVal-minVal),-minVal*255.0/(maxVal-minVal));
                for (int i = 0; i < boxes.size();i++){
                    char text[256];
                    int id = boxes[i].cate;
                    sprintf(text,"%s %lf%%", class_names[id].c_str(),boxes[i].score*100);
                    char text2[256];
                    double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i]) / 1000;
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
                    cv::rectangle(cv_ptr->image, cv::Point(boxes[i].x1, boxes[i].y1),
                    cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);

                    cv::rectangle(normalizedDepthImage, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);
                    cv::putText(normalizedDepthImage, text2, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    cv::rectangle(normalizedDepthImage, cv::Point(boxes[i].x1, boxes[i].y1),
                    cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
                }
                cv::imshow("normalized depth image", normalizedDepthImage);
                cv::imshow("image", cv_ptr->image);
                cv::waitKey(1);
            }
            if (print) {
            for (int i = 0; i < boxes.size(); i++) {
                double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i]) / 1000;
                std::cout << "x1:" << boxes[i].x1 << ", y1:" << boxes[i].y1 << ", x2:" << boxes[i].x2 << ", y2:" << boxes[i].y2
                << ", conf:" << boxes[i].score << ", id:" << boxes[i].cate << ", " << class_names[boxes[i].cate] << ", dist:" << distance << std::endl;
            }
            }
        }
        
    private:
    //topic, sub and pub param
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
        std_msgs::Float32MultiArray sign_msg;
    //results
        std::vector<TargetBox> boxes;
        std::vector<TargetBox> boxes_depth;
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point stop;
        std::chrono::microseconds duration;

        std::vector<double> depths;
        std::vector<float> confidence_thresholds;
        std::vector<float> distance_thresholds;
        std::vector<std::string> class_names;
    
    //engine inference related
        
        std::vector<std::vector<std::vector<float>>> featureVectors; //output memory
        std::vector<std::vector<cv::cuda::GpuMat>> inputs; //input memory
        cv::cuda::GpuMat img;
        // std::vector<nvinfer1::Dims3>& inputDims;
        std::vector<int> inputDims;
        std::vector<int> outputDims;
        //auto& inputDims;
        bool succ;
        size_t batchSize;
        cv::cuda::GpuMat resized;
        std::vector<cv::cuda::GpuMat> input;
        
        
        

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

int main(int argc, char** argv) {
  int opt;

  // Initialize ROS node and publisher
    ros::init(argc, argv, "object_detector");
  ros::NodeHandle nh;
  
    
    options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing calibration data.
    options.calibrationDataDirectoryPath = "";
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;
    // model(options);
    std::cout << "ck0" << std::endl;
  signTRT signTRT(nh);
  //define rate
  ros::Rate loop_rate(25);

  // Spin ROS node
  while (ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}