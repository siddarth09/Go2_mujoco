# convex_mpc/com_traj.py
from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.signal import cont2discrete
from scipy.linalg import expm

from robot_data import PinGo2ModelMJCF
from foot_step_planner import Gait


class ComTraj:
    """
    Centroidal reference + time-varying linear dynamics for convex MPC.

    State (matches PinGo2ModelMJCF.get_centroidal_state):
        x = [p_com(3), rpy(3), v_com_world(3), omega_body(3)] ∈ R^12

    Forces u are WORLD-frame:
        u = [f_FL_world(3), f_FR_world(3), f_RL_world(3), f_RR_world(3)] ∈ R^12
    """

    LEG_ORDER = ["FL", "FR", "RL", "RR"]

    def __init__(self, go2: PinGo2ModelMJCF):
        self.dummy_go2 = PinGo2ModelMJCF(go2.mjcf_path)

        # Persistent "next" lever arms (COM->foot) in WORLD
        self.r_fl_next = np.zeros(3)
        self.r_fr_next = np.zeros(3)
        self.r_rl_next = np.zeros(3)
        self.r_rr_next = np.zeros(3)

        # References
        self.pos_traj_world = None
        self.rpy_traj_world = None
        self.vel_traj_world = None
        self.omega_traj_body = None

        # Dynamics
        self.Ac = None
        self.Bc = None
        self.gc = None
        self.Ad = None
        self.Bd = None
        self.gd = None

        self.contact_table = None
        self.N = 0

        # Useful scalars for MPC bounds/debug
        self.mass = None
        self.initial_x = None

    # ---------------- Helpers ----------------
    @staticmethod
    def _skew(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float).reshape(3)
        return np.array(
            [[0.0, -r[2], r[1]],
             [r[2], 0.0, -r[0]],
             [-r[1], r[0], 0.0]],
            dtype=float,
        )

    @staticmethod
    def _integrate_g(A: np.ndarray, g: np.ndarray, dt: float) -> np.ndarray:
        tau = np.linspace(0.0, dt, 64)
        exp_terms = [expm(A * t) @ g for t in tau]
        return np.trapz(np.stack(exp_terms, axis=1), tau, axis=1).reshape(-1, 1)

    @staticmethod
    def _rpy_to_R_body_to_world(rpy: np.ndarray) -> np.ndarray:
        R = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        return np.asarray(R, dtype=float)

    def compute_x_ref_vec(self) -> np.ndarray:
        if self.pos_traj_world is None:
            raise RuntimeError("compute_x_ref_vec() called before generate_traj().")
        return np.vstack([
            self.pos_traj_world,
            self.rpy_traj_world,
            self.vel_traj_world,
            self.omega_traj_body,
        ])

    # ---------------- Main API ----------------
    def generate_traj(
        self,
        go2: PinGo2ModelMJCF,
        gait: Gait,
        time_now: float,
        vx_body: float,
        vy_body: float,
        z_des: float,
        yaw_rate_body: float,
        dt: float,
    ) -> None:
        # --- current centroidal state
        x0 = go2.get_centroidal_state()
        pcom0 = x0[0:3].copy()
        rpy0  = x0[3:6].copy()
        self.initial_x = x0.copy()

        # --- mass & centroidal inertia (Pinocchio)
        m = float(go2.data.Ig.mass)
        I_world = np.asarray(go2.data.Ig.inertia, dtype=float)  # expressed in WORLD-aligned frame
        self.mass = m

        # --- horizon length
        T = float(gait.gait_period) if hasattr(gait, "gait_period") else float(1.0 / gait.hz)
        N = max(1, int(np.round(T / dt)))
        self.N = N
        t_vec = (np.arange(N) + 1) * dt

        # --- velocity ref: body -> world using current base rotation
        R_bw_now = go2.get_base_rotation_body_to_world()
        v_world = R_bw_now @ np.array([vx_body, vy_body, 0.0], dtype=float)

        # --- position ref
        self.pos_traj_world = np.zeros((3, N), dtype=float)
        self.pos_traj_world[0, :] = pcom0[0] + v_world[0] * t_vec
        self.pos_traj_world[1, :] = pcom0[1] + v_world[1] * t_vec
        self.pos_traj_world[2, :] = z_des

        # --- vel ref (world)
        self.vel_traj_world = np.tile(v_world.reshape(3, 1), (1, N))

        # --- rpy ref (world angles), omega ref stored in BODY
        self.rpy_traj_world = np.zeros((3, N), dtype=float)
        self.rpy_traj_world[0, :] = rpy0[0]
        self.rpy_traj_world[1, :] = rpy0[1]
        self.rpy_traj_world[2, :] = rpy0[2] + yaw_rate_body * t_vec

        self.omega_traj_body = np.zeros((3, N), dtype=float)
        self.omega_traj_body[2, :] = yaw_rate_body

        # --- contact table as (4, N)
        if hasattr(gait, "compute_contact_table"):
            try:
                ct = gait.compute_contact_table(time_now, dt, N)
            except TypeError:
                ct = gait.compute_contact_table(time_now, N)
        else:
            ct = np.zeros((4, N), dtype=int)
            for k in range(N):
                ct[:, k] = np.asarray(gait.compute_current_mask(time_now + k * dt), dtype=int)

        ct = np.asarray(ct)
        if ct.shape == (N, 4):
            ct = ct.T
        if ct.shape != (4, N):
            raise ValueError(f"contact_table must be shape (4,N); got {ct.shape}")
        self.contact_table = ct.astype(int)

        # --- lever arm trajectories (WORLD): r_i = p_foot - p_com
        r_fl = np.zeros((3, N), dtype=float)
        r_fr = np.zeros((3, N), dtype=float)
        r_rl = np.zeros((3, N), dtype=float)
        r_rr = np.zeros((3, N), dtype=float)

        # init "next" levers from current measured state (WORLD)
        self.r_fl_next = go2.get_foot_lever_world("FL")
        self.r_fr_next = go2.get_foot_lever_world("FR")
        self.r_rl_next = go2.get_foot_lever_world("RL")
        self.r_rr_next = go2.get_foot_lever_world("RR")

        # mask tracking (assume gait mask: 1=stance, 0=swing)
        mask_prev = np.asarray(gait.compute_current_mask(time_now), dtype=int).reshape(4)

        # dummy rollout joints
        q_joints0  = go2.state.joint_pos.copy()
        dq_joints0 = go2.state.joint_vel.copy()

        # helper to set/get per-leg next vector
        def _get_next(leg: str) -> np.ndarray:
            return getattr(self, f"r_{leg.lower()}_next")

        def _set_next(leg: str, r_next: np.ndarray) -> None:
            setattr(self, f"r_{leg.lower()}_next", r_next)

        for k in range(N):
            # rotation from rpy reference (body->world)
            rpy_k = self.rpy_traj_world[:, k]
            R_bw_k = self._rpy_to_R_body_to_world(rpy_k)
            R_wb_k = R_bw_k.T

            # build dummy q,dq for touchdown heuristics
            base_pos_k = self.pos_traj_world[:, k].copy()
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

            q = np.concatenate([base_pos_k, quat_xyzw, q_joints0]).astype(float)

            v_body_k = R_wb_k @ self.vel_traj_world[:, k]
            w_body_k = np.array([0.0, 0.0, self.omega_traj_body[2, k]], dtype=float)
            dq = np.concatenate([v_body_k, w_body_k, dq_joints0]).astype(float)

            self.dummy_go2.update(q, dq)

            # mask at this step
            mask = np.asarray(gait.compute_current_mask(time_now + k * dt), dtype=int).reshape(4)

            # UPDATE foothold at TOUCHDOWN: swing -> stance (0 -> 1)
            for i, leg in enumerate(self.LEG_ORDER):
                if (mask_prev[i] == 0) and (mask[i] == 1):
                    # touchdown world position from gait, compute new lever arm to dummy COM
                    p_td = gait.compute_touchdown_world(self.dummy_go2, leg)
                    p_com = self.dummy_go2.get_com_position()
                    r_next = (p_td - p_com).astype(float)

                    _set_next(leg, r_next)

                    # also update the convenience attributes used elsewhere
                    if leg == "FL": self.r_fl_next = r_next
                    if leg == "FR": self.r_fr_next = r_next
                    if leg == "RL": self.r_rl_next = r_next
                    if leg == "RR": self.r_rr_next = r_next

            # Fill r_i(k). For MPC, only stance matters (swing forces are constrained ~0).
            # We hold lever arm at the planned next foothold during stance.
            for i, leg in enumerate(self.LEG_ORDER):
                r_next = _get_next(leg)
                if leg == "FL":
                    r_fl[:, k] = r_next if mask[i] == 1 else (r_fl[:, k-1] if k > 0 else r_next)
                elif leg == "FR":
                    r_fr[:, k] = r_next if mask[i] == 1 else (r_fr[:, k-1] if k > 0 else r_next)
                elif leg == "RL":
                    r_rl[:, k] = r_next if mask[i] == 1 else (r_rl[:, k-1] if k > 0 else r_next)
                elif leg == "RR":
                    r_rr[:, k] = r_next if mask[i] == 1 else (r_rr[:, k-1] if k > 0 else r_next)

            mask_prev = mask

        self.r_fl_foot_world = r_fl
        self.r_fr_foot_world = r_fr
        self.r_rl_foot_world = r_rl
        self.r_rr_foot_world = r_rr

        # --- Continuous dynamics
        # x = [p, rpy, v_world, omega_body]
        self.Ac = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3),        np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        ]).astype(float)

        self.Bc = np.zeros((N, 12, 12), dtype=float)

        for k in range(N):
            rpy_k = self.rpy_traj_world[:, k]
            R_bw_k = self._rpy_to_R_body_to_world(rpy_k)
            R_wb_k = R_bw_k.T

            # Inertia should be consistent with omega_body:
            # If I_world is world-aligned, convert to body: I_body = R_wb I_world R_bw
            I_body = R_wb_k @ I_world @ R_bw_k
            I_inv_body = np.linalg.inv(I_body)

            r_world_list = [r_fl[:, k], r_fr[:, k], r_rl[:, k], r_rr[:, k]]

            # v_dot (WORLD) = (1/m) Σ f_world + g
            B_v = (1.0 / m) * np.tile(np.eye(3), (1, 4))  # 3x12

            # omega_dot (BODY) = I_body^{-1} Σ ( r_body × f_body )
            # f_body = R_wb * f_world, r_body = R_wb * r_world
            B_w_blocks = []
            for r_w in r_world_list:
                r_b = R_wb_k @ r_w
                B_w_blocks.append(I_inv_body @ self._skew(r_b) @ R_wb_k)  # 3x3
            B_w = np.hstack(B_w_blocks)  # 3x12

            self.Bc[k] = np.block([
                [np.zeros((3, 12))],
                [np.zeros((3, 12))],
                [B_v],
                [B_w],
            ])

        # gravity in v_dot (WORLD z up -> g = -9.81 in z)
        self.gc = np.array([0, 0, 0,
                            0, 0, 0,
                            0, 0, -9.81,
                            0, 0, 0], dtype=float).reshape(12,)

        # --- Discretize (Ad constant, Bd(k) varying)
        self.Bd = np.zeros_like(self.Bc)
        for k in range(N):
            Ad, Bd_k, _, _, _ = cont2discrete(
                (self.Ac, self.Bc[k], np.eye(12), np.zeros((12, 12))),
                dt,
            )
            self.Ad = np.asarray(Ad, dtype=float)
            self.Bd[k] = np.asarray(Bd_k, dtype=float)

        self.gd = self._integrate_g(self.Ac, self.gc, dt)
