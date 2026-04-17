import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import numpy as np
import random


class QLearning:
    def __init__(self):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.5
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.q_table = np.zeros((50, 3))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])

        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state, action]
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class LearningNode(Node):
    def __init__(self):
        super().__init__('learning_node')

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.qlearn = QLearning()
        self.state = 0

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = ranges[np.isfinite(ranges)]

        if len(ranges) == 0:
            return

        min_distance = np.min(ranges)

        state = int(min_distance * 10)
        state = max(0, min(state, 49))  # limiter entre 0 et 49

        action = self.qlearn.choose_action(state)

        cmd = Twist()

        # Actions
        if action == 0:
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
        elif action == 1:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif action == 2:
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5

        # Reward
        if min_distance < 0.3:
            reward = -1
        else:
            reward = 1

        # Update Q-learning
        self.qlearn.update(self.state, action, reward, state)

        # AFFICHAGE IMPORTANT 
        print(f"State: {state} | Action: {action} | Reward: {reward}")
        print("Q-table snapshot:")
        print(self.qlearn.q_table)

        # Update state
        self.state = state

        # Send command
        self.publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = LearningNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
