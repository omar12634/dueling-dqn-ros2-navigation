import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

import numpy as np

from .DQN import DQNAgent


class DQNNode(Node):

    def __init__(self):

        super().__init__('dqn_learning_node')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(1.0, self.control_loop)

        self.state_size = 10
        self.action_size = 3

        self.agent = DQNAgent(self.state_size, self.action_size)

        self.get_logger().info("DQN Learning Started")


    def control_loop(self):

        state = np.random.rand(self.state_size)

        action = self.agent.act(state)

        msg = Twist()

        if action == 0:
            msg.linear.x = 0.2
            msg.angular.z = 0.0

        elif action == 1:
            msg.linear.x = 0.0
            msg.angular.z = 0.5

        elif action == 2:
            msg.linear.x = 0.0
            msg.angular.z = -0.5

        self.publisher.publish(msg)

        self.get_logger().info(f"Action chosen: {action}")


def main(args=None):

    rclpy.init(args=args)

    node = DQNNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
