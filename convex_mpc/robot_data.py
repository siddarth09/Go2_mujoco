#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pinocchio as pin


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def quat_xyzw_to_rpy(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → roll-pitch-yaw."""
    x, y, z, w = q_xyzw
    R = pin.Quaternion(w, x, y, z).toRotationMatrix()
    return pin.rpy.matrixToRpy(R)


def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    """roll-pitch-yaw -> quat (x,y,z,w)."""
    R = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
    q = pin.Quaternion(R)  # returns w,x,y,z
    return np.array([q.x, q.y, q.z, q.w], dtype=float)


# ---------------------------------------------------------
# ConfigurationState (reference-style)
# ---------------------------------------------------------
@dataclass
class ConfigurationState:
    """
    Stores the robot configuration in a readable form and can assemble:
      q  ∈ R^(7+12)  = [p(3), quat_xyzw(4), joints(12)]
      dq ∈ R^(6+12) = [v_body(3), w_body(3), joint_vels(12)]

    Notes:
      - quat stored as (x,y,z,w) to match your existing convention.
      - dq base linear velocity is assumed BODY-frame (common with Pinocchio FF).
    """
    base_pos: np.ndarray          # (3,)
    base_quat_xyzw: np.ndarray    # (4,)  [x,y,z,w]
    joint_pos: np.ndarray         # (12,)

    base_lin_vel_body: np.ndarray # (3,)
    base_ang_vel_body: np.ndarray # (3,)
    joint_vel: np.ndarray         # (12,)

    @staticmethod
    def zeros() -> "ConfigurationState":
        return ConfigurationState(
            base_pos=np.zeros(3),
            base_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            joint_pos=np.zeros(12),
            base_lin_vel_body=np.zeros(3),
            base_ang_vel_body=np.zeros(3),
            joint_vel=np.zeros(12),
        )

    def get_q(self) -> np.ndarray:
        return np.concatenate([self.base_pos, self.base_quat_xyzw, self.joint_pos]).astype(float)

    def get_dq(self) -> np.ndarray:
        return np.concatenate([self.base_lin_vel_body, self.base_ang_vel_body, self.joint_vel]).astype(float)

    def set_from_q(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        self.base_pos = q[0:3].copy()
        self.base_quat_xyzw = q[3:7].copy()
        self.joint_pos = q[7:].copy()

    def set_from_dq(self, dq: np.ndarray) -> None:
        dq = np.asarray(dq, dtype=float).reshape(-1)
        self.base_lin_vel_body = dq[0:3].copy()
        self.base_ang_vel_body = dq[3:6].copy()
        self.joint_vel = dq[6:].copy()


# ---------------------------------------------------------
# High-level Go2 model (reference-style)
# ---------------------------------------------------------
class PinGo2ModelMJCF:
    """
    Reference-equivalent model wrapper built from MJCF:
      - Loads model
      - Maintains state
      - Updates Pinocchio
      - Exposes COM, feet, Jacobians, dynamics, centroidal state
    """

    LEG_ORDER = ["FL", "FR", "RL", "RR"]

    def __init__(self, mjcf_path: str):
        self.mjcf_path = str(Path(mjcf_path).resolve())

        # --- Build Pinocchio model from MJCF with floating base ---
        self.model = pin.buildModelFromMJCF(
            self.mjcf_path,
            root_joint=pin.JointModelFreeFlyer()
        )
        self.data = self.model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv

        # --- Frame names from your MJCF convention ---
        # Feet: calf bodies (Pinocchio frames exist for bodies)
        self.foot_frames: Dict[str, str] = {
            "FL": "FL_calf",
            "FR": "FR_calf",
            "RL": "RL_calf",
            "RR": "RR_calf",
        }
        self.foot_frame_ids = {leg: self.model.getFrameId(name) for leg, name in self.foot_frames.items()}

        # Hips: hip bodies (used to compute hip offsets)
        self.hip_frames: Dict[str, str] = {
            "FL": "FL_hip",
            "FR": "FR_hip",
            "RL": "RL_hip",
            "RR": "RR_hip",
        }
        self.hip_frame_ids = {leg: self.model.getFrameId(name) for leg, name in self.hip_frames.items()}

        # Base frame (often exists as a body frame; if not, we use the model root)
        self.base_frame_name = "base"
        self.base_frame_id = self.model.getFrameId(self.base_frame_name) if self.base_frame_name in self.model.frames else None

        # State + yaw unwrap
        self.state = ConfigurationState.zeros()
        self.prev_yaw = 0.0

        # Computed / cached per update
        self.R_body_to_world = np.eye(3)
        self.R_world_to_body = np.eye(3)
        self.base_rpy = np.zeros(3)

        # Hip offsets (in base frame) computed after first update
        self.hip_offsets_base: Dict[str, np.ndarray] = {leg: np.zeros(3) for leg in self.LEG_ORDER}
        self._hip_offsets_initialized = False

        # Leg joint index ranges (assumes each leg has 3 joints in order in q)
        # If your MJCF ordering differs, you can override these indices.
        # Default: [FL(0:3), FR(3:6), RL(6:9), RR(9:12)] in the JOINT portion.
        self.leg_joint_slices = {
            "FL": slice(0, 3),
            "FR": slice(3, 6),
            "RL": slice(6, 9),
            "RR": slice(9, 12),
        }

    # -----------------------------------------------------
    # Core update
    # -----------------------------------------------------
    def update(self, q: np.ndarray, dq: np.ndarray) -> None:
        """
        Update kinematics & dynamics. q,dq must follow:
          q  = [p(3), quat_xyzw(4), joints(12)]
          dq = [v_body(3), w_body(3), joint_vels(12)]
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        dq = np.asarray(dq, dtype=float).reshape(-1)

        # store readable state
        self.state.set_from_q(q)
        self.state.set_from_dq(dq)

        # Update pinocchio pipelines
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q, dq)

        # Cache base rotations / rpy
        self._update_base_rotation_and_rpy(q)

        # Initialize hip offsets once
        if not self._hip_offsets_initialized:
            self._compute_hip_offsets_in_base()
            self._hip_offsets_initialized = True

    # -----------------------------------------------------
    # Base quantities
    # -----------------------------------------------------
    def _update_base_rotation_and_rpy(self, q: np.ndarray) -> None:
        quat_xyzw = q[3:7]
        # Convert (x,y,z,w) -> Rotation
        rpy = quat_xyzw_to_rpy(quat_xyzw)

        # unwrap yaw
        dyaw = rpy[2] - self.prev_yaw
        if dyaw > np.pi:
            rpy[2] -= 2 * np.pi
        elif dyaw < -np.pi:
            rpy[2] += 2 * np.pi
        self.prev_yaw = rpy[2]

        self.base_rpy = rpy

        # Rotation body->world from quaternion
        x, y, z, w = quat_xyzw
        R = pin.Quaternion(w, x, y, z).toRotationMatrix()
        self.R_body_to_world = R.copy()
        self.R_world_to_body = R.T.copy()

    def get_base_rpy(self) -> np.ndarray:
        return self.base_rpy.copy()

    def get_base_rotation_body_to_world(self) -> np.ndarray:
        return self.R_body_to_world.copy()

    def get_base_rotation_world_to_body(self) -> np.ndarray:
        return self.R_world_to_body.copy()

    def get_base_angular_velocity_body(self) -> np.ndarray:
        # dq layout: [v_body(0:3), w_body(3:6), joints...]
        return self.state.base_ang_vel_body.copy()

    # -----------------------------------------------------
    # COM / centroidal state
    # -----------------------------------------------------
    def get_com_position(self) -> np.ndarray:
        return self.data.com[0].copy()

    def get_com_velocity(self) -> np.ndarray:
        return self.data.vcom[0].copy()

    def get_centroidal_state(self) -> np.ndarray:
        """
        x = [p_com(3), rpy(3), v_com(3), omega_body(3)] ∈ R^12
        This matches the common centroidal MPC state layout.
        """
        p_com = self.get_com_position()
        v_com = self.get_com_velocity()
        rpy = self.get_base_rpy()
        omega = self.get_base_angular_velocity_body()
        return np.concatenate([p_com, rpy, v_com, omega]).astype(float)

    # -----------------------------------------------------
    # Hip offsets (base frame)
    # -----------------------------------------------------
    def _compute_hip_offsets_in_base(self) -> None:
        """
        Computes hip positions relative to base frame.
        This is useful for Raibert-style foot placement:
          p_hip_world = p_base_world + R * hip_offset_base
        """
        # If base frame exists as a frame, use it; else approximate as floating base origin.
        if self.base_frame_id is not None and self.base_frame_id != 0:
            T_base_world = self.data.oMf[self.base_frame_id]
            R_wb = T_base_world.rotation
            p_wb = T_base_world.translation
        else:
            # floating base origin is q[0:3] and R from quaternion
            R_wb = self.R_body_to_world
            p_wb = self.state.base_pos

        R_bw = R_wb.T

        for leg in self.LEG_ORDER:
            hid = self.hip_frame_ids[leg]
            T_hip_world = self.data.oMf[hid]
            p_wh = T_hip_world.translation
            # hip offset in base frame
            self.hip_offsets_base[leg] = (R_bw @ (p_wh - p_wb)).copy()

    def get_hip_offset_base(self, leg: str) -> np.ndarray:
        return self.hip_offsets_base[leg].copy()

    def get_hip_position_world(self, leg: str) -> np.ndarray:
        """
        p_hip_world = p_base + R * hip_offset_base
        """
        return self.state.base_pos + self.R_body_to_world @ self.get_hip_offset_base(leg)

    # -----------------------------------------------------
    # Foot quantities
    # -----------------------------------------------------
    def get_foot_position_world(self, leg: str) -> np.ndarray:
        fid = self.foot_frame_ids[leg]
        return self.data.oMf[fid].translation.copy()

    def get_all_foot_positions_world(self) -> Dict[str, np.ndarray]:
        return {leg: self.get_foot_position_world(leg) for leg in self.LEG_ORDER}

    def get_foot_lever_world(self, leg: str) -> np.ndarray:
        """
        r = p_foot - p_com (world frame)
        Used in centroidal torque: τ = r × f
        """
        return self.get_foot_position_world(leg) - self.get_com_position()

    def get_foot_velocity_world(self, q: np.ndarray, dq: np.ndarray, leg: str) -> np.ndarray:
        """
        v_foot_world = J_world(q) * dq
        """
        J = self.get_full_foot_jacobian_world(q, leg)  # 3xnv
        return (J @ dq).reshape(3)

    # -----------------------------------------------------
    # Jacobians
    # -----------------------------------------------------
    def get_full_foot_jacobian_world(self, q: np.ndarray, leg: str) -> np.ndarray:
        """
        Linear Jacobian (3 x nv), WORLD frame:
          v_foot_world = J * dq
        """
        fid = self.foot_frame_ids[leg]
        J6 = pin.computeFrameJacobian(
            self.model, self.data, q, fid, pin.ReferenceFrame.WORLD
        )  # 6 x nv
        return J6[:3, :].copy()

    def get_full_foot_jacobian_body(self, q: np.ndarray, leg: str) -> np.ndarray:
        """
        Linear Jacobian (3 x nv), LOCAL (frame) coordinates.
        """
        fid = self.foot_frame_ids[leg]
        J6 = pin.computeFrameJacobian(
            self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL
        )
        return J6[:3, :].copy()

    def get_leg_foot_jacobian_world_3x3(self, q: np.ndarray, leg: str) -> np.ndarray:
        """
        Returns 3x3 Jacobian wrt only that leg's 3 joints.
        Assumes joint ordering in the joint portion is:
          FL(0:3), FR(3:6), RL(6:9), RR(9:12)
        """
        J_full = self.get_full_foot_jacobian_world(q, leg)  # 3 x nv
        # nv = 6 (freeflyer) + 12 (joints) => joints start at column 6
        js = self.leg_joint_slices[leg]
        if isinstance(js, slice):
            if js.start is None or js.stop is None:
                raise ValueError(f"Leg joint slice for {leg} must have start/stop, got {js}")
            cols = slice(6 + js.start, 6 + js.stop, js.step)  # shift by 6 for floating base
        elif isinstance(js, (list, tuple, np.ndarray)):
            cols = [6 + int(i) for i in js]
        else:
            cols = 6 + int(js)

        return J_full[:, cols].copy()
        

    def get_leg_foot_jacobian_body_3x3(self, q: np.ndarray, leg: str) -> np.ndarray:
        J_full = self.get_full_foot_jacobian_body(q, leg)
        js = self.leg_joint_slices[leg]
        return J_full[:, 6 + js].copy()

    def get_leg_Jdot_dq_world(self, q: np.ndarray, dq: np.ndarray, leg: str) -> np.ndarray:
        """
        Computes Jdot*dq (linear part) in world frame for the foot frame.
        Useful in operational-space accel control:
          pdd = J*qdd + Jdot*dq
        """
        fid = self.foot_frame_ids[leg]
        a6 = pin.getFrameClassicalAcceleration(
            self.model, self.data, fid, pin.ReferenceFrame.WORLD
        )  # spatial acceleration (classic) of the frame origin
        # classical acceleration relates to Jdot*dq when qdd=0:
        # a = J*qdd + Jdot*dq  => if qdd=0, a = Jdot*dq
        return a6.linear.copy()

    # -----------------------------------------------------
    # Dynamics terms
    # -----------------------------------------------------
    def mass_matrix(self) -> np.ndarray:
        return self.data.M.copy()

    def coriolis_matrix(self) -> np.ndarray:
        return self.data.C.copy()

    def gravity(self) -> np.ndarray:
        return self.data.g.copy()


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    mjcf = "/home/sid/projects25/src/Go2_mujoco/unitree_go2/go2.xml"

    robot = PinGo2ModelMJCF(mjcf)

    # Pinocchio neutral state
    q = pin.neutral(robot.model)  # (nq,)
    dq = np.zeros(robot.nv)

    # Our convention expects quat as xyzw in q[3:7].
    # pin.neutral provides the quaternion in its own convention (for free-flyer it's usually [x y z qw qx qy qz]?).
    # So for safety, we overwrite with identity quaternion in xyzw:
    q = q.copy()
    q[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

    robot.update(q, dq)

    print("Centroidal state x (12D):")
    print(robot.get_centroidal_state())

    print("\nHip offsets (base frame):")
    for leg in robot.LEG_ORDER:
        print(leg, robot.get_hip_offset_base(leg))

    print("\nFoot positions (world):")
    for leg, p in robot.get_all_foot_positions_world().items():
        print(leg, p)

    print("\nMass matrix shape:", robot.mass_matrix().shape)
