import time
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Float64MultiArray
from numpy import float64
import numpy as np

from ur_msgs.srv import SetIO


class Ur5JointController(Node):
    def __init__(self, v_cm=35, f_update=120):  # Max v = 35 cm/s
        super().__init__("position_control_node")
        # self.lock = threading.Lock()
        self.d_t = 1 / f_update  # Time between updates
        # Check if max speed is within limits (longest link = 44 cm)
        if v_cm > 35:
            self.v_cm = 0.35
            self.get_logger().warn(
                "Max speed is too high. Setting Max speed v_cm = 25 cm/s"
            )

        self.d_phi = np.float64(
            v_cm * (1 / f_update) / 44
        )  # Max angle delta per update

        # Read the current joint positions from the joint state topic
        self.traj_controller_state: JointTrajectoryControllerState = None  # type: ignore
        self.state_subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )
        # Create a publisher to send joint position commands to the UR5 robot
        self.forward_pos_pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )
        # Create a subscriber to read the agents control commands
        self.joint_cmd = self.create_subscription(
            Float64MultiArray, "/joint_cmd", self.receive_joint_delta, 10
        )

        # Delta list to store the most recent control command
        self.current_angle_delta: list[float64] = [np.float64(0.0)] * 6
        # Current joint positions as received from the joint state topic
        self.live_joint_positions: list[float64] | None = None
        self.live_joint_velocities: list[float64] | None = None
        self.live_joint_torques: list[float64] | None = None
        # Set a ground truth of the robot state to avoid drift
        self.joint_positions_GT: list[float64] | None = None

        # Set the initial joint positions to a default pose  TODO (implement reset to home position)
        self.init_joint_positions: list[float] = [
            0.0,  # shoulder_pan_joint
            -1.6,  # shoulder_lift_joint
            1.6,  # elbow_joint
            -3.0,  # wrist_1_joint
            -1.7,  # wrist_2_joint
            0.0,  # wrist_3_joint
        ]

        # Define the joint names and order
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Set update rate for the joint command
        self.update_timer = self.create_timer(self.d_t, self.send_joint_command)

        # Gripper init
        self.cli = self.create_client(SetIO, "/io_and_status_controller/set_io")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for gripper /set_io service...")
        self.get_logger().info("/set_io service is available.")

        self.current_gripper_state = False  # 0 = closed, 1 = open
        self.gripper_target = False

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg
        # Create a dict to sort the values in defined order
        name_val_mapping: dict[str, float64] = dict(zip(msg.name, msg.position))  # type: ignore
        # Get the joint positions in the order of joint_names
        self.live_joint_positions = [
            name_val_mapping[joint]
            for joint in self.joint_names  # use dict to get the values in the order of joint_names
        ]

        vel_name_val_mapping: dict[str, float64] = dict(zip(msg.name, msg.velocity))  # type: ignore

        self.live_joint_velocities = [
            vel_name_val_mapping[joint]
            for joint in self.joint_names  # use dict to get the values in the order of joint_names
        ]

        torque_name_val_mapping: dict[str, float64] = dict(zip(msg.name, msg.effort))  # type: ignore

        self.live_joint_torques = [
            torque_name_val_mapping[joint]
            for joint in self.joint_names  # use dict to get the values in the order of joint_names
        ]

        # Set the ground truth joint positions if not set yet
        if self.joint_positions_GT is None and self.live_joint_positions is not None:
            self.joint_positions_GT = self.live_joint_positions.copy()

    def get_joint_positions(self) -> list[float64]:
        """_summary_
        Function to return the current joint positions of the UR5 robot when called.
        """
        # If no joint positions received yet, return None
        if self.live_joint_positions is None:
            self.get_logger().warn("No joint positions received yet")
            return None  # type: ignore
        gripper_state = np.float64(-1.0 if self.current_gripper_state == 0.0 else 1.0)
        all_joint_positions = self.live_joint_positions + [(gripper_state)]
        return all_joint_positions

    def get_joint_observation(self) -> dict | None:
        obs = {
            "joint_positions": self.live_joint_positions,
            "joint_velocities": self.live_joint_velocities,
            "joint_torques": self.live_joint_torques,
            "gripper_state": self.current_gripper_state,
        }

        # Check if any of the values are None
        if any(value is None for value in obs.values()):
            return None
        return obs

    def receive_joint_delta(self, msg: Float64MultiArray):
        """_summary_
        Function to receive the joint delta command from a ros msg.
        """
        normalized_delta: list[float] = msg.data  # type: ignore
        self.set_joint_delta(normalized_delta)
        # log the received command

    def set_joint_delta(self, normalized_delta: list[float]):
        """_summary_
        Function to set the joint delta from a normalized list of values.
        """
        # Check if the received data has the correct length
        if len(normalized_delta) != 7:
            self.get_logger().warn("Received invalid joint delta command")
            return
        # Denormalize the angles (first 6 elements)
        angle_delta = [norm_val * self.d_phi for norm_val in normalized_delta[:6]]

        self.current_angle_delta = angle_delta
        self.gripper_target = (
            1
            if normalized_delta[6] > 0
            else 0  # negative values open the gripper (D_out=0), positive values close it (D_out=1)
        )

    def check_drift(self):
        """_summary_
        Function to check if the robot has drifted from the ground truth state.
        """
        if (
            self.joint_positions_GT is not None
            and self.live_joint_positions is not None
        ):
            # Calculate the difference between the current joint positions and the ground truth
            diff = [
                live - GT
                for live, GT in zip(self.live_joint_positions, self.joint_positions_GT)
            ]
            # If the difference is larger than a threshold, reset the robot
            if sum(diff) > 0.005:
                self.get_logger().warn("Robot has drifted. Resetting the ground truth")
                self.joint_positions_GT = self.live_joint_positions.copy()

    def send_joint_command(self, duration: float = 0):
        """_summary_
        Takes the stored (most recent) command values and sends them to the UR5 robot.
        Called at a fixed rate by the update_timer.
        """
        # Check if the robot has drifted from the ground truth state
        self.check_drift()

        # If no duration is provided, use the default duration
        if duration == 0:
            duration = self.d_t
        # If no joint positions received yet abort
        if self.joint_positions_GT is None:
            self.get_logger().warn("Waiting for the urs joint positions")
            return
        # Set the gripper to the target state
        if self.current_gripper_state != self.gripper_target:
            self.set_gripper(self.gripper_target)  # type: ignore
            self.current_gripper_state = self.gripper_target

        if sum(self.current_angle_delta) != 0:
            # Calculate the new target joint positions by adding the delta to the current joint positions
            new_target = [
                delta + position
                for delta, position in zip(
                    self.current_angle_delta, self.joint_positions_GT
                )
            ]

            # Enforce joint limits
            new_target = self.enforce_joint_limits(new_target)

            # Create a ForwardPositionController message
            new_target_msg = Float64MultiArray()
            new_target_msg.data = new_target

            # Publish the target joint state message
            self.forward_pos_pub.publish(new_target_msg)
            self.current_angle_delta = [np.float64(0.0)] * 6
            self.joint_positions_GT = new_target

    def set_gripper(self, state: bool):
        req = SetIO.Request()
        req.fun = 1  # Digital output
        req.pin = 16  # Gripper close pin
        req.state = float(state)
        # Call the service asynchronously and handle the response in the callback
        future = self.cli.call_async(req)
        future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future):
        try:
            pass  # Log the gripper response if needed
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def enforce_joint_limits(self, target_angles: list[float64]):
        """Safety layer to avoid damaging the robot by exceeding joint limits."""
        # TODO  Implement joint limits
        new_target_angles = target_angles
        return new_target_angles

    def reset(self):
        """_summary_
        Function to reset the UR5 robot to its initial joint position.
        """
        self.get_logger().info("Resetting UR5 to initial joint position")
        self.send_joint_command(self.init_joint_positions, duration=3)  # type: ignore


def main(args=None):
    rclpy.init(args=args)
    node = Ur5JointController()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
