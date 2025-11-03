#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time


class Go2JointCommandTest(Node):
    def __init__(self):
        super().__init__('go2_joint_command_test')

        # Publisher to the position controller
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/go2_position_controller/commands',
            10
        )

        # Timer to periodically publish commands
        self.timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Total joints: 12 (4 legs × 3 joints)
        self.num_joints = 12

        # Motion parameters
        self.phase = 0.0
        self.amplitude = 0.5  # radians
        self.speed = 0.5      # Hz

        self.get_logger().info('Go2 random/sine joint command publisher started.')

    def timer_callback(self):
        msg = Float64MultiArray()

        # Option 1 – random values within safe range:
        # positions = np.random.uniform(-1.0, 1.0, self.num_joints)

        # Option 2 – smooth sine-wave motion:
        positions = self.amplitude * np.sin(
            np.linspace(0, 2*np.pi, self.num_joints) + self.phase
        )

        msg.data = positions.tolist()
        self.publisher_.publish(msg)

        self.get_logger().info(f'Published: {np.round(positions, 3)}')

        # Increment phase for smooth motion
        self.phase += 2 * np.pi * self.speed * self.timer_period
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi


def main(args=None):
    rclpy.init(args=args)
    node = Go2JointCommandTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
