import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import numpy as np
from rl_turtlbot.Qlearning import QLearning


class LearningNode(Node):

    def __init__(self):

        super().__init__('learning_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        self.qlearn = QLearning()

        self.state = 0

        self.get_logger().info("RL Node Started")


    def scan_callback(self, msg):

        ranges = np.array(msg.ranges)

        ranges = ranges[np.isfinite(ranges)]

        if len(ranges) == 0:
            return

        min_distance = np.min(ranges)

        state = int(min_distance * 10)

        action = self.qlearn.choose_action(state)

        twist = Twist()

        if action == 0:
            twist.linear.x = 0.2
            twist.angular.z = 0.0

        elif action == 1:
            twist.linear.x = 0.0
            twist.angular.z = 0.5

        elif action == 2:
            twist.linear.x = 0.0
            twist.angular.z = -0.5

        if min_distance < 0.3:
            reward = -1
        else:
            reward = 1

        self.qlearn.update_q_table(self.state, action, reward, state)

        self.state = state

        self.cmd_pub.publish(twist)

        print("Distance:", min_distance)
        print("Q-table:")
        print(self.qlearn.q_table)


def main(args=None):

    rclpy.init(args=args)

    node = LearningNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
