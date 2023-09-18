#! /usr/bin/env python3
import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped

def publish_static_transforms(static_broadcaster):

    # Create an empty list to hold the transforms
    static_transforms = []

    #Add transform from chassis to wheel_rear_left
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()

    # Create TransformStamped for chassis -> camera
    t_camera = TransformStamped()
    t_camera.header.stamp = rospy.Time.now()
    t_camera.header.frame_id = "chassis"
    t_camera.child_frame_id = "camera"
    # Joint pose offset
    joint_x, joint_y, joint_z = 0, 0, 0
    # Link pose
    link_x, link_y, link_z = 0, 0, 0.2
    # Combined pose
    t_camera.transform.translation.x = joint_x + link_x
    t_camera.transform.translation.y = joint_y + link_y
    t_camera.transform.translation.z = joint_z + link_z
    
    # Euler angles from pose
    roll, pitch, yaw = 0, 0.2617, 0

    # Convert Euler angles to quaternion
    qtn = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    # Set the rotation values in the TransformStamped message
    t_camera.transform.rotation.x = qtn[0]
    t_camera.transform.rotation.y = qtn[1]
    t_camera.transform.rotation.z = qtn[2]
    t_camera.transform.rotation.w = qtn[3]
    # Add to the list of static transforms
    static_transforms.append(t_camera)

    # Create TransformStamped for chassis -> laser
    t_laser = TransformStamped()
    t_laser.header.stamp = rospy.Time.now()
    t_laser.header.frame_id = "chassis"
    t_laser.child_frame_id = "laser"
    # Joint pose offset
    joint_x, joint_y, joint_z = 0, 0, 0
    # Link pose
    link_x, link_y, link_z = 0, 0, 0.2
    # Combined pose
    t_laser.transform.translation.x = joint_x + link_x
    t_laser.transform.translation.y = joint_y + link_y
    t_laser.transform.translation.z = joint_z + link_z

    # Set the rotation values in the TransformStamped message
    t_laser.transform.rotation.x = 0
    t_laser.transform.rotation.y = 0
    t_laser.transform.rotation.z = 0
    t_laser.transform.rotation.w = 1
    # Add to the list of static transforms
    static_transforms.append(t_laser)

    ## Create TransformStamped for chassis -> steer_left
    # t_steer_left = TransformStamped()
    # t_steer_left.header.stamp = rospy.Time.now()
    # t_steer_left.header.frame_id = "chassis"
    # t_steer_left.child_frame_id = "steer_left"
    # # Joint pose offset
    # joint_x, joint_y, joint_z = 0, 0, 0.005
    # # Link pose
    # link_x, link_y, link_z = 0.112, 0.0725, 0
    # # Combined pose
    # t_steer_left.transform.translation.x = joint_x + link_x
    # t_steer_left.transform.translation.y = joint_y + link_y
    # t_steer_left.transform.translation.z = joint_z + link_z
    # # No rotation 
    # t_steer_left.transform.rotation.x = 0.0
    # t_steer_left.transform.rotation.y = 0.0
    # t_steer_left.transform.rotation.z = 0.0
    # t_steer_left.transform.rotation.w = 1.0
    # # Add to the list of static transforms
    # static_transforms.append(t_steer_left)

    # # Create TransformStamped for chassis -> steer_right
    # t_steer_right = TransformStamped()
    # t_steer_right.header.stamp = rospy.Time.now()
    # t_steer_right.header.frame_id = "chassis"
    # t_steer_right.child_frame_id = "steer_right"
    # # Joint pose offset
    # joint_x, joint_y, joint_z = 0, 0, -0.005
    # # Link pose
    # link_x, link_y, link_z = 0.112, -0.0725, 0
    # # Combined pose
    # t_steer_right.transform.translation.x = joint_x + link_x
    # t_steer_right.transform.translation.y = joint_y + link_y
    # t_steer_right.transform.translation.z = joint_z + link_z
    # # No rotation
    # t_steer_right.transform.rotation.x = 0.0
    # t_steer_right.transform.rotation.y = 0.0
    # t_steer_right.transform.rotation.z = 0.0
    # t_steer_right.transform.rotation.w = 1.0
    # # Add to the list of static transforms
    # static_transforms.append(t_steer_right)

    # # Create TransformStamped for chassis -> wheel_rear_left
    # t_wheel_rear_left = TransformStamped()
    # t_wheel_rear_left.header.stamp = rospy.Time.now()
    # t_wheel_rear_left.header.frame_id = "chassis"
    # t_wheel_rear_left.child_frame_id = "wheel_rear_left"
    # # Joint pose offset
    # joint_x, joint_y, joint_z = 0, 0, 0
    # # Link pose
    # link_x, link_y, link_z = -0.152, 0.081, 0
    # # Combined pose
    # t_wheel_rear_left.transform.translation.x = joint_x + link_x
    # t_wheel_rear_left.transform.translation.y = joint_y + link_y
    # t_wheel_rear_left.transform.translation.z = joint_z + link_z
    # # No rotation
    # t_wheel_rear_left.transform.rotation.x = 0.0
    # t_wheel_rear_left.transform.rotation.y = 0.0
    # t_wheel_rear_left.transform.rotation.z = 0.0
    # t_wheel_rear_left.transform.rotation.w = 1.0
    # # Add to the list of static transforms
    # static_transforms.append(t_wheel_rear_left)

    # # Create TransformStamped for chassis -> wheel_rear_right
    # t_wheel_rear_right = TransformStamped()
    # t_wheel_rear_right.header.stamp = rospy.Time.now()
    # t_wheel_rear_right.header.frame_id = "chassis"
    # t_wheel_rear_right.child_frame_id = "wheel_rear_right"
    # # Joint pose offset
    # joint_x, joint_y, joint_z = 0, 0, 0
    # # Link pose
    # link_x, link_y, link_z = -0.152, -0.081, 0
    # # Combined pose
    # t_wheel_rear_right.transform.translation.x = joint_x + link_x
    # t_wheel_rear_right.transform.translation.y = joint_y + link_y
    # t_wheel_rear_right.transform.translation.z = joint_z + link_z
    # # No rotation
    # t_wheel_rear_right.transform.rotation.x = 0.0
    # t_wheel_rear_right.transform.rotation.y = 0.0
    # t_wheel_rear_right.transform.rotation.z = 0.0
    # t_wheel_rear_right.transform.rotation.w = 1.0
    # # Add to the list of static transforms
    # static_transforms.append(t_wheel_rear_right)

    # # Create TransformStamped for chassis -> wheel_front_left
    # t_wheel_front_left = TransformStamped()
    # t_wheel_front_left.header.stamp = rospy.Time.now()
    # t_wheel_front_left.header.frame_id = "chassis"
    # t_wheel_front_left.child_frame_id = "wheel_front_left"
    # # Joint pose offset
    # joint_x, joint_y, joint_z = 0, 0, 0
    # # Link pose
    # link_x, link_y, link_z = 0.118, 0.081, 0
    # # Combined pose
    # t_wheel_front_left.transform.translation.x = joint_x + link_x
    # t_wheel_front_left.transform.translation.y = joint_y + link_y
    # t_wheel_front_left.transform.translation.z = joint_z + link_z
    # # No rotation
    # t_wheel_front_left.transform.rotation.x = 0.0
    # t_wheel_front_left.transform.rotation.y = 0.0
    # t_wheel_front_left.transform.rotation.z = 0.0
    # t_wheel_front_left.transform.rotation.w = 1.0
    # # Add to the list of static transforms
    # static_transforms.append(t_wheel_front_left)

    # # Create TransformStamped for chassis -> wheel_front_right
    # t_wheel_front_right = TransformStamped()
    # t_wheel_front_right.header.stamp = rospy.Time.now()
    # t_wheel_front_right.header.frame_id = "chassis"
    # t_wheel_front_right.child_frame_id = "wheel_front_right"
    # # Joint pose offset
    # joint_x, joint_y, joint_z = 0, 0, 0
    # # Link pose
    # link_x, link_y, link_z = 0.118, -0.081, 0
    # # Combined pose
    # t_wheel_front_right.transform.translation.x = joint_x + link_x
    # t_wheel_front_right.transform.translation.y = joint_y + link_y
    # t_wheel_front_right.transform.translation.z = joint_z + link_z
    # # No rotation
    # t_wheel_front_right.transform.rotation.x = 0.0
    # t_wheel_front_right.transform.rotation.y = 0.0
    # t_wheel_front_right.transform.rotation.z = 0.0
    # t_wheel_front_right.transform.rotation.w = 1.0
    # # Add to the list of static transforms
    # static_transforms.append(t_wheel_front_right)

    # Publish all static transforms
    static_broadcaster.sendTransform(static_transforms)

if __name__ == '__main__':
    rospy.init_node('static_tf2_broadcaster')
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        publish_static_transforms(static_broadcaster)
        rate.sleep()
