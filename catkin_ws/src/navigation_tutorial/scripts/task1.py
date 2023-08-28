#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
class MoveToEdges:
    def __init__(self):
        rospy.init_node('move_to_edges_node', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.map_data = None
        self.robot_pose = None
    def map_callback(self, map_msg):
        self.map_data = map_msg
    def odom_callback(self, odom_msg):
        orientation = odom_msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.robot_pose = odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw
    def get_map_value(self, x, y):
        if self.map_data is not None:
            width = self.map_data.info.width
            index = y * width + x
            return self.map_data.data[index]
        return -1
    def move_to_position(self, target_x, target_y):
        while self.map_data is None or self.robot_pose is None:
            rospy.loginfo("Waiting for map and robot pose...")
            self.rate.sleep()
        while not rospy.is_shutdown():
            x, y, _ = self.robot_pose
            map_value = self.get_map_value(int(x), int(y))
            if map_value == 0:  # If robot is on a free cell
                twist = Twist()
                twist.linear.x = 0.2  # Move forward
                twist.angular.z = 0.0  # Stop rotation
                self.cmd_vel_pub.publish(twist)
            else:
                twist = Twist()
                twist.linear.x = 0.0  # Stop
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
            # Check if robot reached the target position
            if math.isclose(x, target_x, abs_tol=0.1) and math.isclose(y, target_y, abs_tol=0.1):
                twist = Twist()
                twist.linear.x = 0.0  # Stop
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                rospy.loginfo("Robot reached the target position.")
                break
            self.rate.sleep()
    def move_to_edges(self):
        self.move_to_position(0.0, self.map_data.info.height - 1)  # Right upper corner
        self.move_to_position(self.map_data.info.width - 1, self.map_data.info.height - 1)  # Left upper corner
if __name__ == '__main__':
    try:
        mover = MoveToEdges()
        mover.move_to_edges()
    except rospy.ROSInterruptException:
        pass