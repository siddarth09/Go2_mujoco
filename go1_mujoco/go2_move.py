import mujoco
import mujoco.viewer
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class LegInfo:
    hip_joint: str    # abduction
    thigh_joint: str  # hip pitch
    calf_joint: str   # knee
    foot_site: str    # site at the foot tip

class GoMove:
    def __init__(self, xml_path: str):
        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Map legs to their joints/sites from your XML
        self.legs = {
            "FL": LegInfo("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot"),
            "FR": LegInfo("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot"),
            "RL": LegInfo("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot"),
            "RR": LegInfo("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot"),
        }

        # Reset to keyframe "home" so we start in a stable pose
        self._reset_to_keyframe("home")
        mujoco.mj_forward(self.model, self.data)

        # Capture per-leg stand targets from home pose
        self._capture_stand_targets_from_home()

        # Build joint index mapping
        self.jidx = {}
        for j in range(self.model.njnt):
            name = self.model.joint(j).name
            qpos_adr = self.model.jnt_qposadr[j]
            qvel_adr = self.model.jnt_dofadr[j]
            self.jidx[name] = {'qpos': qpos_adr, 'qvel': qvel_adr}

        # Controller gains
        self.Kp = 200.0
        self.Kd = 5.0

        # Link lengths (approximate)
        self.L1 = 0.213
        self.L2 = 0.213

    # ------------------
    # Low-level helpers
    # ------------------

    def _reset_to_keyframe(self, key_name: str):
        """Reset sim state to a named keyframe (like 'home')."""
        key_id = self.model.key(key_name).id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        # also forward to make sure transforms are valid
        mujoco.mj_forward(self.model, self.data)

    def step_sim(self, ctrl):
        """
        Advance the simulation one step with given control vector.
        ctrl should be same length as model.nu (number of actuators).
        """
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def get_joint_angles(self, joint_names):
        """Return current angles q for a list of joint names."""
        return np.array([
            self.data.qpos[self.jidx[name]]
            for name in joint_names
        ])

    def get_joint_velocities(self, joint_names):
        """Return current qdot for a list of joint names."""
        return np.array([
            self.data.qvel[self.jidx[name]]
            for name in joint_names
        ])

    def world_foot_position(self, leg_key: str):
        """Return 3D world position of the given leg's foot site."""
        site_name = self.legs[leg_key].foot_site
        site_id = self.model.site(site_name).id
        return np.array(self.data.site_xpos[site_id])

    # --------------------------------
    # IK for a single leg (approximate)
    # --------------------------------

    def leg_ik(self, foot_target_in_hip, leg_key: str):
        """
        Solve IK for one leg.
        foot_target_in_hip: np.array([x,y,z]) in that leg's hip frame.
        Returns np.array([q_hip_abd, q_hip_pitch, q_knee_pitch])

        NOTE: This is a first-cut geometric IK. We'll refine sign
        conventions per leg as needed when we test.
        """
        x, y, z = foot_target_in_hip

        # 1) abduction: rotate about +X or +Y depending on frame.
        # We'll start with a simple guess:
        q_abd = np.arctan2(y, -z)  # heuristic lateral placement

        # project into sagittal plane after "removing" abduction effect.
        # For first pass we'll ignore coupling and just solve planar IK in x-z.
        # distance from hip to foot in 2D plane
        d = np.sqrt(x**2 + z**2)

        # Law of cosines for knee
        cos_knee = (d**2 - self.L1**2 - self.L2**2) / (2*self.L1*self.L2)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        knee_inner = np.arccos(cos_knee)
        # We want a bending knee. For typical quadrupeds:
        q_knee = -(np.pi - knee_inner)

        # Hip pitch angle
        # angle from hip to target:
        phi = np.arctan2(-z, x)
        # angle of second link wrt first:
        beta = np.arctan2(self.L2 * np.sin(np.pi - knee_inner),
                          self.L1 + self.L2 * np.cos(np.pi - knee_inner))
        q_hip = phi - beta

        return np.array([q_abd, q_hip, q_knee])

    # --------------------------------------------------
    # Build a full-body pose (all 4 legs) from foot goals
    # --------------------------------------------------

    def solve_full_pose(self, foot_targets):
        """
        foot_targets: dict like
           {
             "FL": np.array([x,y,z]),
             "FR": np.array([x,y,z]),
             "RL": np.array([x,y,z]),
             "RR": np.array([x,y,z]),
           }
        Returns:
           q_des_dict = {
             "FL_hip_joint":  angle,
             "FL_thigh_joint":angle,
             "FL_calf_joint": angle,
             ...
           }
        """
        q_des = {}
        for leg_key, target in foot_targets.items():
            # compute IK for that leg
            q_leg = self.leg_ik(target, leg_key)

            leginfo = self.legs[leg_key]
            q_des[leginfo.hip_joint]   = q_leg[0]
            q_des[leginfo.thigh_joint] = q_leg[1]
            q_des[leginfo.calf_joint]  = q_leg[2]

        return q_des

    # --------------------------------------------------
    # PD control towards a desired joint dictionary
    # --------------------------------------------------

    def pd_control(self, q_des_dict):
        """Produce actuator commands (ctrl) using PD in joint space."""
        u = np.zeros(self.model.nu)
        for act_id in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[act_id][0]
            joint_name = self.model.joint(jnt_id).name

            q = self.data.qpos[self.jidx[joint_name]['qpos']]
            qd = self.data.qvel[self.jidx[joint_name]['qvel']]

            q_des = q_des_dict.get(joint_name, q)
            qd_des = 0.0

            err = q_des - q
            derr = qd_des - qd
            u[act_id] = self.Kp * err + self.Kd * derr

        return u

    # --------------------------------------------------
    # Convenience poses
    # --------------------------------------------------

    def world_to_body(self, body_name, p_world):
        """
        Convert a 3D point in world coordinates into the local frame of a given body.
        Compatible with MuJoCo >= 3.1.
        """
        bid = self.model.body(body_name).id
        body_pos_w = self.data.xpos[bid]             # (3,)
        body_rot_w = self.data.xmat[bid].reshape(3, 3)  # world-from-body
        R_bw = body_rot_w.T  # body-from-world
        return R_bw @ (p_world - body_pos_w)

    
    def _capture_stand_targets_from_home(self):
        """
        After we're in 'home' keyframe, record each foot's position
        expressed in that leg's hip frame. This becomes our nominal stand pose.
        """
        self.stand_targets_local = {}

        for leg_key, leginfo in self.legs.items():
            # 1. world position of the foot site
            site_id = self.model.site(leginfo.foot_site).id
            foot_world = np.array(self.data.site_xpos[site_id])  # (3,)

            # 2. express that foot position in that leg's hip body frame
            #    hip body names in your XML are FL_hip, FR_hip, RL_hip, RR_hip
            hip_body_name = leg_key + "_hip"
            foot_in_hip = self.world_to_body(hip_body_name, foot_world)

            self.stand_targets_local[leg_key] = foot_in_hip.copy()

    def desired_feet_stand(self):
        """
        Return the nominal per-leg foot target (in each hip frame)
        captured from the 'home' keyframe. This should match a realistic stand.
        """
        # Just return a copy to avoid accidental mutation
        return {
            leg_key: self.stand_targets_local[leg_key].copy()
            for leg_key in self.legs.keys()
        }

    def desired_feet_sit(self):
        """
        Create a crouched/sit pose by pulling the feet 'up' in each hip frame.
        We'll move them closer by reducing |z| and |x|.
        """
        targets = {}
        for leg_key, base_target in self.stand_targets_local.items():
            t = base_target.copy()

            # bring feet closer up to the body:
            # - shorten x (fore/aft reach)
            # - raise z (less downward extension)
            t[0] *= 0.9   # pull slightly in toward hip in x
            t[2] *= 1.4   # 40% closer vertically (less negative magnitude)

            targets[leg_key] = t
        return targets

    # --------------------------------------------------
    # High-level motion: interpolate between 2 poses
    # --------------------------------------------------

    def move_between_poses(self, q_start_dict, q_goal_dict, duration=1.0, dt=0.002, viewer=None):
        """
        Interpolate joints from q_start_dict -> q_goal_dict over 'duration' seconds.
        If a viewer is passed, reuses it. If None, runs headless.
        """
        steps = int(duration / dt)

        for i in range(steps):
            if viewer is not None and not viewer.is_running():
                break

            alpha = i / max(1, steps - 1)
            blended = {}
            for jname in q_goal_dict.keys():
                q0 = q_start_dict.get(jname, q_goal_dict[jname])
                q1 = q_goal_dict[jname]
                blended[jname] = (1.0 - alpha) * q0 + alpha * q1

            ctrl = self.pd_control(blended)
            self.step_sim(ctrl)

            if viewer is not None:
                viewer.sync()
                time.sleep(dt)


    # --------------------------------------------------
    # Helpers to capture current joint targets (for interpolation start)
    # --------------------------------------------------

    def current_joint_dict(self):
        """
        Snapshot of current joint angles, keyed by joint name,
        for just our controlled joints.
        """
        out = {}
        for leg_key, leginfo in self.legs.items():
            for jname in [leginfo.hip_joint, leginfo.thigh_joint, leginfo.calf_joint]:
                out[jname] = float(self.data.qpos[self.jidx[jname]['qpos']])
        return out



if __name__ == "__main__":
   
    robot = GoMove("/home/siddarth/manipulation_ws/src/go1_mujoco/unitree_go2/scene_mjx.xml")

    # Reset to home so you see it when viewer opens
    robot._reset_to_keyframe("home")
    mujoco.mj_forward(robot.model, robot.data)

    # Open viewer once and keep it open
    v = mujoco.viewer.launch_passive(robot.model, robot.data)
    print("✅ Viewer opened — starting in HOME pose")

    stand_targets = robot.desired_feet_stand()
    q_stand = robot.solve_full_pose(stand_targets)

    sit_targets = robot.desired_feet_sit()
    q_sit = robot.solve_full_pose(sit_targets)

    q_now = robot.current_joint_dict()

    # Wait a bit before motion starts
    time.sleep(2.0)

    print("➡️ Moving to STAND pose...")
    robot.move_between_poses(q_now, q_stand, duration=2.0, dt=0.002, viewer=v)
    print("✅ Reached stand pose")

    time.sleep(1.0)

    print("➡️ Moving to SIT pose...")
    robot.move_between_poses(q_stand, q_sit, duration=2.5, dt=0.002, viewer=v)
    print("✅ Reached sit pose")

    # Wait 5 seconds sitting
    time.sleep(5.0)

    print("➡️ Returning to STAND pose...")
    robot.move_between_poses(q_sit, q_stand, duration=2.5, dt=0.002, viewer=v)
    print("✅ Returned to stand pose")

    print("🎬 Motion sequence complete — close the window manually when done.")
    while v.is_running():
        v.sync()
        time.sleep(0.01)
