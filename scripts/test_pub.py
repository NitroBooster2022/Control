#! /usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np

if __name__ == "__main__":
    #2.初始化 ROS 节点:命名(唯一)
    rospy.init_node("talker_p")
    #3.实例化 发布者 对象
    pub = rospy.Publisher("automobile/command",String,queue_size=3)
    #4.组织被发布的数据，并编写逻辑发布数据
    msg = String()  #创建 msg 对象
    msg_front = "hello 你好"
    count = 0  #计数器 
    control_commands = np.load("/home/simonli/Documents/Simulator/src/control/scripts/mpc_u_c.npy")

    # 设置循环频率
    speed1 = 0.2
    speed2 = -0.2
    steering_angle = 0
    speed = speed1
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if count>=30:
            count = 0
            speed = speed1 if speed < 0 else speed2
            rospy.loginfo("toggle speed to %f",speed)
        # speed = control_commands[count,0]
        # steering_angle = control_commands[count,1]
        msg.data = '{"action":"1","speed":'+str(speed)+'}'
        pub.publish(msg)
        # msg.data = '{"action":"2","steerAngle":'+str(float(steering_angle*180/np.pi))+'}'
        # pub.publish(msg)
        rate.sleep()
        count += 1