#include "ros/ros.h"
#include "utils/localisation.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "std_msgs/Header.h"
#include "tf2/LinearMath/Quaternion.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class LocalizationBridge {
    public:
        LocalizationBridge(ros::NodeHandle& nh_):
            nh(nh_)
        {
            nh.getParam("/max_noise", max_noise);
            pose_pub = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("/gps", 10);
            sub = nh_.subscribe("/automobile/localisation", 10, &LocalizationBridge::localisationCallback, this);
            std::fill(std::begin(msg.pose.covariance), std::end(msg.pose.covariance), 0.0);
            // σ^2 = (b - a)^2 / 12
            float pos_cov = std::pow(2*max_noise, 2) / 12;
            ROS_INFO("Position covariance: %f", pos_cov);
            msg.pose.covariance[0] = pos_cov; // Variance for posA
            msg.pose.covariance[7] = pos_cov; // Variance for posB
            msg.pose.covariance[14] = pos_cov; // Variance for z position (set to 0.1 or 0 if not used)
            msg.pose.covariance[21] = pos_cov; // Variance for rotA (roll)
            msg.pose.covariance[28] = pos_cov; // Variance for rotB (pitch)
            msg.pose.covariance[35] = pos_cov;
        }
    private:
        ros::NodeHandle nh;
        ros::Publisher pose_pub;
        ros::Subscriber sub;
        double max_noise;
        // Create a PoseWithCovarianceStamped message
        geometry_msgs::PoseWithCovarianceStamped msg;

        void localisationCallback(const utils::localisation::ConstPtr& msg_in) {
            // Copy the header
            msg.header.stamp = msg_in->header.stamp;
            msg.header.frame_id = "odom";

            // Copy the position and orientation data
            msg.pose.pose.position.x = msg_in->posA;
            msg.pose.pose.position.y = 15.0-msg_in->posB;
            msg.pose.pose.position.z = 0.0; // 2d

            tf2::Quaternion q;
            q.setRPY(0, 0, msg_in->rotA); 
            msg.pose.pose.orientation = tf2::toMsg(q);

            // Publish the message
            pose_pub.publish(msg);
        }
};

int main(int argc, char **argv) {
    // Initialize the ROS node
    ros::init(argc, argv, "localization_bridge");
    ros::NodeHandle nh;

    LocalizationBridge bridge(nh);
    while (ros::ok()) {
        ros::spinOnce();
    }

    return 0;
}
