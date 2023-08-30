#!/usr/bin/env python3
import rospy
from navigation_tutorial.srv import MoveTrigger, MoveTriggerRequest

class Task1Manager:
    def __init__(self):
        rospy.init_node('task_manager')
        
        # service client
        self.move_robot = rospy.ServiceProxy('/move_robot', MoveTrigger)

        self.move_robot.wait_for_service()
        rospy.loginfo("Service /move_robot is ready!")

    def move(self, straight, turn):
        command = MoveTriggerRequest()
        command.straight = straight
        command.turn = turn
        rospy.loginfo(f"request: straight={straight}, turn={turn}")
        response = self.move_robot(command) # service call
        return response.success
    

    def main(self):
        """
        タスクの流れを手続的に記述する
        """
        result = self.move(2.7, 0.0)
        if result:
          rospy.loginfo("move success!")
        else:
          rospy.loginfo("move failed!")
        
        result = self.move(0.0, 90.0)
        if result:
          rospy.loginfo("turn success!")
        else:
          rospy.loginfo("turn failed!")

        result = self.move(2.5, 0.0)
        if result:
          rospy.loginfo("move success!")
        else:
          rospy.loginfo("move failed!")

        result = self.move(0.0, -90.0)
        if result:
          rospy.loginfo("turn success!")
        else:
          rospy.loginfo("turn failed!")

if __name__ == '__main__':
    task_manager = Task1Manager()
    task_manager.main()
