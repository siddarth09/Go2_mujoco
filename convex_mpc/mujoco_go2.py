#!/usr/bin/env python3
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from pathlib import Path
import pinocchio as pin

from robot_data import PinGo2ModelMJCF


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
SCENE_XML = Path("/home/sid/projects25/src/Go2_mujoco/unitree_go2/scene.xml")


class MujocoGo2Model:
    def __init__(self, scene_xml: Path = SCENE_XML):
        scene_xml = Path(scene_xml).expanduser().resolve()

        self.model = mj.MjModel.from_xml_path(str(scene_xml))
        self.data = mj.MjData(self.model)

        # Validate base body
        self.base_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        if self.base_body_id < 0:
            raise RuntimeError("MuJoCo: could not find body named 'base_link'")

        # Cache actuator ids for torque control
        self.actuator_ids = {}
        for leg in ["FL", "FR", "RL", "RR"]:
            aids = [
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_hip"),
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_thigh"),
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_calf"),
            ]
            if any(a < 0 for a in aids):
                raise RuntimeError(
                    f"MuJoCo: actuator name mismatch for leg {leg}. "
                    f"Expected actuators: {leg}_hip, {leg}_thigh, {leg}_calf"
                )
            self.actuator_ids[leg] = aids

        # Optional: sanity check dimensions for Go2 free joint + 12 joints
        if self.model.nq != 19:
            print(f"[WARN] model.nq={self.model.nq} (expected 19 for free joint + 12 joints)")
        if self.model.nv != 18:
            print(f"[WARN] model.nv={self.model.nv} (expected 18 for 6D base + 12 joint vels)")

    # --------------------------------------------------
    # Pinocchio → MuJoCo
    # --------------------------------------------------
    def set_state_from_pin(self, q_pin: np.ndarray):
        """
        Pinocchio q (your convention):
          q_pin = [px py pz qx qy qz qw q1..q12]   (len 19)

        MuJoCo qpos expects free joint as:
          qpos = [px py pz qw qx qy qz q1..q12]    (len 19)
        """
        q_pin = np.asarray(q_pin, dtype=float).reshape(-1)
        if q_pin.shape[0] != self.model.nq:
            raise ValueError(f"q_pin has len {q_pin.shape[0]} but mujoco model.nq={self.model.nq}")

        px, py, pz = q_pin[:3]
        qx, qy, qz, qw = q_pin[3:7]
        joints = q_pin[7:]

        # IMPORTANT: MuJoCo freejoint quaternion order is (w, x, y, z)
        self.data.qpos[:] = np.concatenate(
            [[px, py, pz, qw, qx, qy, qz], joints]
        )

        # update derived quantities
        mj.mj_forward(self.model, self.data)

    # --------------------------------------------------
    # MuJoCo → Pinocchio
    # --------------------------------------------------
    def update_pin_from_mu(self, go2_pin: PinGo2ModelMJCF):
        """
        Sync Pinocchio model using MuJoCo state.

        MuJoCo:
          qpos: [p(3), quat_wxyz(4), joints(12)]
          qvel: [v_world(3), w_body(3), joints(12)]   (MuJoCo convention for free joint)

        Pinocchio wrapper expects:
          q_pin:  [p(3), quat_xyzw(4), joints(12)]
          dq_pin: [v_body(3), w_body(3), joints(12)]
        """
        # ensure kinematics are consistent before reading qpos/qvel
        mj.mj_forward(self.model, self.data)

        q_mj = np.asarray(self.data.qpos, dtype=float).copy()
        dq_mj = np.asarray(self.data.qvel, dtype=float).copy()

        # MuJoCo quaternion in qpos is (w, x, y, z)
        qw, qx, qy, qz = q_mj[3:7]

        # Rotation body -> world
        R_wb = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()

        # MuJoCo stores base linear velocity in WORLD frame for the free joint
        v_world = dq_mj[0:3]
        # MuJoCo stores base angular velocity in BODY frame
        w_body = dq_mj[3:6]

        # Pinocchio free-flyer velocity convention (in your pipeline) is BODY-frame linear velocity
        v_body = R_wb.T @ v_world

        # Build Pinocchio-format q/dq (quat as x,y,z,w)
        q_pin = np.concatenate([q_mj[0:3], [qx, qy, qz, qw], q_mj[7:]])
        dq_pin = np.concatenate([v_body, w_body, dq_mj[6:]])

        go2_pin.update(q_pin, dq_pin)

    # --------------------------------------------------
    # Torques
    # --------------------------------------------------
    def set_joint_torques(self, tau: np.ndarray):
        """
        tau ∈ R^12 ordered as:
        [FL_hip, FL_thigh, FL_calf,
         FR_hip, FR_thigh, FR_calf,
         RL_hip, RL_thigh, RL_calf,
         RR_hip, RR_thigh, RR_calf]
        """
        tau = np.asarray(tau, dtype=float).reshape(-1)
        if tau.shape[0] != 12:
            raise ValueError(f"tau must be len 12, got {tau.shape[0]}")

        idx = 0
        for leg in ["FL", "FR", "RL", "RR"]:
            aids = self.actuator_ids[leg]
            for j in range(3):
                self.data.ctrl[aids[j]] = tau[idx]
                idx += 1

    def step(self):
        mj.mj_step(self.model, self.data)

    # --------------------------------------------------
    # Viewer
    # --------------------------------------------------
    def launch_viewer(self):
        viewer = mjv.launch_passive(self.model, self.data)

        viewer.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = self.base_body_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        return viewer


if __name__ == "__main__":
    # Pinocchio model (robot only)
    go2_pin = PinGo2ModelMJCF("/home/sid/projects25/src/Go2_mujoco/unitree_go2/go2.xml")

    # MuJoCo model (scene + robot)
    sim = MujocoGo2Model()
    viewer = sim.launch_viewer()

    tau = np.zeros(12)

    # make sure everything is initialized
    mj.mj_forward(sim.model, sim.data)

    while viewer.is_running():
        # Sync Pinocchio with MuJoCo
        sim.update_pin_from_mu(go2_pin)

        # (later) compute MPC torques here
        sim.set_joint_torques(tau)

        sim.step()
        viewer.sync()
