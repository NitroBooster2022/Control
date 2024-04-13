#include "ros/ros.h"
#include "include/yolo-fastestv2.h"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "cv_bridge/cv_bridge.h"
#include "std_msgs/Header.h"
#include "include/engine.h"
#include <chrono>
#include <vector>
#include <std_msgs/Float32MultiArray.h>
#include "yolov8.h"

using namespace nvinfer1;
// Options options;
// Engine model(options);
//get the model path
std::string model_name = "citycocov2lgtclab_20.engine";
// std::string filePathParam = __FILE__;
// size_t pos = __FILE__.rfind("/") + 1;
// std::string modelPath = __FILE__.substr(0,__FILE__.rfind("/")) + "/model/" + model_name;
// std::string modelPath = filePathParam.replace(pos,std::string::npos, "model/" + model_name + ".engine");
std::string modelPath = "/home/slsecret/Documents/Simulator/src/Control/src/model/citycocov2lgtclab_20.onnx";
// std::cout << modelPath << std::endl;
YoloV8Config config;
YoloV8 yolov8 = YoloV8(modelPath, config);


class signTRT{
    public:
        
        // Specify what precision to use for inference
        // FP16 is approximately twice as fast as FP32.
        
        signTRT(ros::NodeHandle& nh):it(nh),reshapedOutput(1,std::vector<std::vector<float>>(8400, std::vector<float>(17)))
        // scores(10),classes(10),tmp_boxes(10)//, inputDims(inputDims)
        {
            
            //loading from yaml file
            nh.getParam("class_names",class_names);
            nh.getParam("confidence_thresholds", confidence_thresholds);
            nh.getParam("max_distance_thresholds", distance_thresholds);
            nh.getParam("/signFastest/showFlag",show);
            nh.getParam("/signFastest/printFlag",print);
            nh.getParam("signFastest/printDuration",printDuration);
            
            // nh.getParam("model",model_name);
            std::cout << "showFlag: " << show << std::endl;
            std::cout << "printFlag: " << print << std::endl;
            std::cout << "printDuration: " << printDuration << std::endl;
            std::cout << "class_names: " << class_names.size() << std::endl;
            std::cout << "confidence_thresholds: " << confidence_thresholds.size() << std::endl;
            // std::cout << "distance_thresholds: " << distance_thresholds.size() << std::endl;
            std::cout << "model: " << model_name << std::endl;
            
            // filePathParam.replace(pos,std::string::npos, "model/" + model_name + ".engine");
            // const char* modelPath = filePathParam.c_str();
            //input image dimension
            // inputDims = {model.getInputDims()[0].d[0],model.getInputDims()[0].d[1]};
            inputDims = {640,640};
            //initialize engine
            
            // engine(options); //create engine object
            // model.m_engineName = filePathParam;
            // yoloV8.loadEngine(filePathParam);
            std::cout << "ck1" << std::endl;

            
            // Load the TensorRT engine file from disk
            // succ = model.loadNetwork();
            // if (!succ) {
            //     throw std::runtime_error("Unable to load TRT engine.");
            // }
            // std::cout << "ck2" << std::endl;
            //topics management
            pub = nh.advertise<std_msgs::Float32MultiArray>("sign",10);
            std::cout << "pub created" << std::endl;
            
            depth_sub = it.subscribe("/camera/depth/image_raw",3,&signTRT::depthCallback,this);
            ros::topic::waitForMessage<sensor_msgs::Image>("/camera/depth/image_raw",nh);
            sub = it.subscribe("/camera/color/image_raw",3,&signTRT::imageCallback,this);

        }
        void depthCallback(const sensor_msgs::ImageConstPtr& msg){
            // std::cout << "dpt callback"<< std::endl;
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
            std::cout << "OpenCV version : " << CV_VERSION << std::endl;
            if(printDuration) start = std::chrono::high_resolution_clock::now();
            // try{
            std::cout<<"ck0"<< std::endl;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            std::cout<<"ck1"<< std::endl;
            std::vector<Object> detected_objects = yolov8.detectObjects(cv_ptr->image);
            // }
            // catch (cv_bridge::Exception& e){
            //     ROS_ERROR("cv_bridge exception: %s",e.what());
            //     return;
            // }
            //input image
            
            std::cout<<"ck2"<< std::endl;
            
            // img.upload(cv_ptr->image);
            // m_imgHeight = img.rows;
            // m_imgWidth = img.cols;
            // m_ratio =  1.f / std::min(inputDims[0]/ static_cast<float>(img.cols), inputDims[0] / static_cast<float>(img.rows));
            
            // cv::cuda::cvtColor(img, resized, cv::COLOR_BGR2RGB);

            

            //detection params
            // inputDims = model.getInputDims();
            // batchSize = options.optBatchSize;
            // std::cout << "batchSize: " << batchSize << std::endl;
            // resized = Engine::resizeKeepAspectRatioPadRightBottom(img,640,480); //inputDims[0].d[1], inputDims[0].d[2])
//             for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
//             if (resized.rows != inputDims[1] || resized.cols != inputDims[2])
//                 // resized = Engine::resizeKeepAspectRatioPadRightBottom(resized, inputDims[1],inputDims[0]);
//                 cv::cuda::resize(resized,resized,cv::Size(inputDims[1],inputDims[0]));
//             input.emplace_back(std::move(resized));
// //                 // std::cout<<input.data()<<std::endl;
// //             }
//             cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
//             cv::Mat img_RGB;
//             resized.download(img_RGB);
//             // Show our image inside it.
//             cv::imshow("Display window",img_RGB);

//             // Wait for a keystroke in the window
//             cv::waitKey(0);

//             inputs.emplace_back(std::move(input));
//             preciseStopwatch s2;
            
//             succ = model.runInference(inputs,featureVectors);

//             static long long t2 = 0;
//             t2 = s2.elapsedTime<long long, std::chrono::microseconds>();
//             std::cout << "Avg Inference time: " << (t2) / 1000.f << " ms" << std::endl;
            preciseStopwatch s3;

//             inputs.clear();
//             //convert featureVectors into output boxes
            
//             // std::cout << featureVectors[0].size()<< std::endl;
//             // outputDims = {model.getOutputDims()[0].d[0],model.getOutputDims()[0].d[1],model.getOutputDims()[0].d[2],model.getOutputDims()[0].d[3]}; //[1]number of channels [2]number of anchors
            
//             // rework 03/24
//             std::cout<<"ck1"<<std::endl;
//             Engine::transformOutput(featureVectors, featureVector);
//             std::cout<<featureVector.size()<<std::endl;

            
//             // std::cout<<featureVector<<std::endl;
//             std::cout<<"ck2"<<std::endl;
//             outputDims = model.getOutputDims();
//             auto numChannels = outputDims[0].d[1];
//             auto numAnchors = outputDims[0].d[2];

//             // std::cout<<numChannels<<std::endl;
//             // std::cout<<numAnchors<<std::endl;

//             auto numClasses = class_names.size();

//             cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
            

//             for(int i = 0; i < 10; ++i) {
//                 for(int j = 0; j < 17; ++j) {
//                     // Use cv::Mat::at to access elements
//                     std::cout << output.at<float>(i, j) << " ";
//                 }
//                 std::cout << std::endl;
//             }

//             output = output.t();
//             std::cout<<"not transposed"<<std::endl;
//             std::cout<<output.size()<<std::endl;
//             std::cout<<output.rows<<std::endl;
//             std::cout<<output.cols<<std::endl;
//             for (int i = 0; i < numAnchors; i++) {
//                 auto rowPtr = output.row(i).ptr<float>();
//                 if (i < 10){
//                     for (int j = 0; j < 17; j++) {
//                         std::cout << rowPtr[j] << " ";
//                     }
//                     std::cout << std::endl;
//                 }
//                 // auto rowPtr = output.at<float>(i, 0).ptr<float>(i);
//                 auto bboxesPtr = rowPtr;
//                 auto scoresPtr = rowPtr + 4;
//                 auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
//                 float score = *maxSPtr;
//                 if (score > 0.01) {
//                     std::cout<<"score: "<<score<<std::endl;
//                     float x = *bboxesPtr++;
//                     float y = *bboxesPtr++;
//                     float w = *bboxesPtr++;
//                     float h = *bboxesPtr;

//                     float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
//                     float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
//                     float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
//                     float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

//                     int label = maxSPtr - scoresPtr;
//                     cv::Rect_<float> bbox;
//                     bbox.x = x0;
//                     bbox.y = y0;
//                     bbox.width = x1 - x0;
//                     bbox.height = y1 - y0;
//                     std::cout<<bbox<<std::endl;
//                     tmp_boxes.push_back(bbox);
//                     classes.push_back(label);
//                     scores.push_back(score);
//                 }
//             }
           

            //NMS suppression
            
            // cv::dnn::NMSBoxesBatched(tmp_boxes, scores, classes, 0.25, 0.45, indices);
            // cv::dnn::NMSBoxes(tmp_boxes, scores, 0.25, 0.45, indices, 0.5);
            
            // for (auto& i : indices) {
            // tmp_box.cate = classes[i];
            // tmp_box.score = scores[i];
            // tmp_box.x1 = tmp_boxes[i].x;
            // tmp_box.y1 = tmp_boxes[i].y;
            // tmp_box.x2 = tmp_boxes[i].x +  tmp_boxes[i].width; 
            // tmp_box.y2 = tmp_boxes[i].y +  tmp_boxes[i].height;
            // // std::cout << tmp_box.x1 << std::endl;
            // // std::cout << tmp_box.y1 << std::endl;
            // // std::cout << tmp_box.x2 << std::endl;
            // // std::cout << tmp_box.y2 << std::endl;
            // boxes.push_back(tmp_box);
            // }

            // std::cout << boxes.size() <<std::endl;
            

            int hsy = 0;
            //what to publish in topic
            // for (const auto& box:boxes) {
            //     int class_id = box.cate;
            //     float confidence = box.score;
            //     if (confidence >= confidence_thresholds[class_id]){
            //         // std::cout << "ck3" <<std::endl;
            //         double distance = computeMedianDepth(cv_ptr_depth->image,box)/1000;
            //         // double distance = 0.7;
            //         std::cout << distance <<std::endl;
            //         if (distance <= distance_thresholds[class_id]){
            //             // std::cout << "ck5" <<std::endl;
            //             sign_msg.data.push_back(box.x1);
            //             sign_msg.data.push_back(box.y1);
            //             sign_msg.data.push_back(box.x2);
            //             sign_msg.data.push_back(box.y2);
            //             sign_msg.data.push_back(distance);
            //             sign_msg.data.push_back(confidence);
            //             sign_msg.data.push_back(static_cast<float>(class_id));
            //             hsy++;
            //         }
            //     }
            // }
            std_msgs::Float32MultiArray sign_msg;
            sign_msg.layout.data_offset = 0;
            for (const struct Object& box:detected_objects) {
                int class_id = box.label;
                float confidence = box.probability;
                if (confidence >= confidence_thresholds[class_id]){
                    std::cout << "ck3" <<std::endl;
                    double distance = computeMedianDepth(cv_ptr_depth->image,box)/1000;
                    // double distance = 0.7;
                    std::cout << "ck3.1" <<std::endl;
                    std::cout << distance <<std::endl;
                    std::cout << distance_thresholds[class_id] <<std::endl;
                    if (distance <= distance_thresholds[class_id]){
                        std::cout << "ck5" <<std::endl;
                        sign_msg.data.push_back(box.rect.x);
                        sign_msg.data.push_back(box.rect.y);
                        sign_msg.data.push_back(box.rect.x + box.rect.width);
                        sign_msg.data.push_back(box.rect.y + box.rect.height);
                        sign_msg.data.push_back(distance);
                        sign_msg.data.push_back(confidence);
                        sign_msg.data.push_back(static_cast<float>(class_id));
                        hsy++;
                    }
                }
            }
            std::cout<<"ck4"<< std::endl;
            // scores.clear();
            // classes.clear();
            // tmp_boxes.clear();

            if(hsy) {
                std_msgs::MultiArrayDimension dim;
                dim.label = "detections";
                dim.size = hsy;
                dim.stride = boxes.size() * 7;
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
                yolov8.drawObjectLabels(cv_ptr->image, detected_objects);
                double maxVal;
                double minVal;
                cv::minMaxIdx(cv_ptr_depth->image,&minVal,&maxVal);
                cv_ptr_depth->image.convertTo(normalizedDepthImage,CV_8U,255.0/(maxVal-minVal),-minVal*255.0/(maxVal-minVal));
                yolov8.drawObjectLabels(normalizedDepthImage, detected_objects);

                // for (int i = 0; i < boxes.size();i++){
                //     char text[256];
                //     int id = boxes[i].cate;
                //     sprintf(text,"%s %lf%%", class_names[id].c_str(),boxes[i].score*100);
                //     char text2[256];
                //     double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i]) / 1000;
                //     sprintf(text2, "%s %.1fm", class_names[id].c_str(), distance);
                //     int baseLine = 0;
                //     cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    
                //     int x = boxes[i].x1;
                //     int y = boxes[i].y1 - label_size.height - baseLine;
                //     if (y < 0)
                //     y = 0;
                //     if (x + label_size.width > cv_ptr->image.cols)
                //     x = cv_ptr->image.cols - label_size.width;
                //     cv::rectangle(cv_ptr->image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                //     cv::Scalar(255, 255, 255), -1);
                //     cv::putText(cv_ptr->image, text, cv::Point(x, y + label_size.height),
                //     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                //     cv::rectangle(cv_ptr->image, cv::Point(boxes[i].x1, boxes[i].y1),
                //     cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);

                //     cv::rectangle(normalizedDepthImage, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                //     cv::Scalar(255, 255, 255), -1);
                //     cv::putText(normalizedDepthImage, text2, cv::Point(x, y + label_size.height),
                //     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                //     cv::rectangle(normalizedDepthImage, cv::Point(boxes[i].x1, boxes[i].y1),
                //     cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 255, 0), 2, 2, 0);
                // }
                cv::imshow("normalized depth image", normalizedDepthImage);
                cv::imshow("image", cv_ptr->image);
                cv::waitKey(1);
            }
            if (print) {
            for (int i = 0; i < detected_objects.size(); i++) {
                // double distance = computeMedianDepth(cv_ptr_depth->image, boxes[i]) / 1000;
                // std::cout << "x1:" << boxes[i].x1 << ", y1:" << boxes[i].y1 << ", x2:" << boxes[i].x2 << ", y2:" << boxes[i].y2
                // << ", conf:" << boxes[i].score << ", id:" << boxes[i].cate << ", " << class_names[boxes[i].cate] << ", dist:" << distance << std::endl;
                double distance = computeMedianDepth(cv_ptr_depth->image, detected_objects[i]) / 1000;
                std::cout << "x1:" << detected_objects[i].rect.x << ", y1:" << detected_objects[i].rect.y << ", x2:" << detected_objects[i].rect.x + detected_objects[i].rect.width << ", y2:" << detected_objects[i].rect.y + detected_objects[i].rect.height
                << ", conf:" << detected_objects[i].probability << ", id:" << detected_objects[i].label << ", " << class_names[detected_objects[i].label] << ", dist:" << distance << std::endl;

            }
            }
            static long long t3 = 0;
            t3 = s3.elapsedTime<long long, std::chrono::microseconds>();
            std::cout << "Avg Post-processing time: " << (t3) / 1000.f << " ms" << std::endl;
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
        // std_msgs::Float32MultiArray sign_msg;
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

        std::vector<cv::Rect> tmp_boxes;
        std::vector<float> scores;
        std::vector<int> classes;
        TargetBox tmp_box;


        std::vector<int> indices;

        size_t class_ind;
        float maxscore;
        std::vector<float>::iterator maxscore_index;
        float m_imgHeight;
        float m_imgWidth;
        float m_ratio;
    
    //engine inference related
        // YoloV8 yoloV8;
        std::vector<std::vector<std::vector<float>>> featureVectors; //output memory
        std::vector<std::vector<std::vector<float>>> reshapedOutput; //1,8400,17
        std::vector<std::vector<cv::cuda::GpuMat>> inputs; //input memory
        cv::cuda::GpuMat img;
        // std::vector<nvinfer1::Dims3>& inputDims;
        std::vector<int> inputDims;
        std::vector<nvinfer1::Dims> outputDims;
        //auto& inputDims;
        bool succ;
        size_t batchSize;
        cv::cuda::GpuMat resized;
        std::vector<cv::cuda::GpuMat> input;
        std::vector<float> featureVector;


        
        

        double computeMedianDepth(const cv::Mat& depthImage, const struct Object& box) {
            
            // Ensure the bounding box coordinates are valid
            int x1 = std::max(0, static_cast<int>(box.rect.x));
            int y1 = std::max(0, static_cast<int>(box.rect.y));
            int x2 = std::min(depthImage.cols, static_cast<int>((box.rect.x) + (box.rect.width)));
            // int y2 = std::min(depthImage.rows, *box.rect.y + *box.rect.height);
            int y2 = std::min(depthImage.rows, static_cast<int>(box.rect.y + box.rect.height));
            
            // Crop the depth image to the bounding box
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
            // std::cout << "compute3" <<std::endl;
            std::sort(depths.begin(), depths.end());

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
  
    
    // options.precision = Precision::FP16;
    // // If using INT8 precision, must specify path to directory containing calibration data.
    // options.calibrationDataDirectoryPath = "";
    // // Specify the batch size to optimize for.
    // options.optBatchSize = 1;
    // // Specify the maximum batch size we plan on running.
    // options.maxBatchSize = 1;
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