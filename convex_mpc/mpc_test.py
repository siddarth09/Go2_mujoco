import os
os.environ["MPLBACKEND"] = "TkAgg"

import time
import numpy as np
from dataclasses import dataclass, field

# ---------------- NEW CLASSES ----------------
from robot_data import PinGo2ModelMJCF
from mujoco_go2 import MujocoGo2Model  
from com_traj import ComTraj
from srbd_mpc import CentroidalMPC
from leg_controller import LegController
from foot_step_planner import Gait

from plot_helper import (
    plot_mpc_result,
    plot_swing_foot_traj,
    plot_full_traj,
    plot_solve_time,
    hold_until_all_fig_closed,
)

# --------------------------------------------------------------------------------
# Parameters (same as your old script)
# --------------------------------------------------------------------------------

INITIAL_X_POS = -5.0
INITIAL_Y_POS = 0.0
RUN_SIM_LENGTH_S = 1.0

RENDER_HZ = 120.0
RENDER_DT = 1.0 / RENDER_HZ
REALTIME_FACTOR = 1.0  # used only for replay pacing below

@dataclass
class BodyCmdPhase:
    t_start: float
    t_end: float
    x_vel: float
    y_vel: float
    z_pos: float
    yaw_rate: float

CMD_SCHEDULE = [
    BodyCmdPhase(0.0, 1.0,  0.7, 0.0, 0.27, 0.0),
    BodyCmdPhase(1.0, 1.5,  0.0, 0.0, 0.27, 0.0),
    BodyCmdPhase(1.5, 3.0,  0.0, 0.3, 0.27, 0.0),
    BodyCmdPhase(3.0, 4.0,  0.0, 0.0, 0.27, 0.0),
    BodyCmdPhase(4.0, 6.0,  0.0, 0.0, 0.27, 2.0),
    BodyCmdPhase(6.0, 6.5,  0.0, 0.0, 0.27, 0.0),
    BodyCmdPhase(6.5, 8.0,  0.6, 0.0, 0.27, 2.0),
    BodyCmdPhase(8.0, 9.0,  0.8, 0.0, 0.27, 0.0),
    BodyCmdPhase(9.0, 10.0, 0.0, 0.0, 0.27, 0.0),
]

GAIT_HZ = 3.0
GAIT_DUTY = 0.6
GAIT_T = 1.0 / GAIT_HZ

x_vel_des_body = 0.0
y_vel_des_body = 0.0
z_pos_des_body = 0.27
yaw_rate_des_body = 0.0

SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ

CTRL_HZ = 200
CTRL_DT = 1.0 / CTRL_HZ

if SIM_HZ % CTRL_HZ != 0:
    raise ValueError(f"SIM_HZ ({SIM_HZ}) must be divisible by CTRL_HZ ({CTRL_HZ}).")
CTRL_DECIM = SIM_HZ // CTRL_HZ

SIM_STEPS = int(RUN_SIM_LENGTH_S * SIM_HZ)
CTRL_STEPS = int(RUN_SIM_LENGTH_S * CTRL_HZ)

MPC_DT = GAIT_T / 16.0
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))

HIP_LIM = 23.7
ABD_LIM = 23.7
KNEE_LIM = 45.43
SAFETY = 0.9

TAU_LIM = SAFETY * np.array([
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
    HIP_LIM, ABD_LIM, KNEE_LIM,
])

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def get_body_cmd(t: float):
    for phase in CMD_SCHEDULE:
        if phase.t_start <= t < phase.t_end:
            return phase.x_vel, phase.y_vel, phase.z_pos, phase.yaw_rate
    return 0.0, 0.0, 0.27, 0.0


def _call_generate_traj(traj: ComTraj,
                        go2: PinGo2ModelMJCF,
                        gait: Gait,
                        time_now_s: float,
                        vx: float, vy: float, z: float, yaw_rate: float,
                        dt: float):
    """
    Your older ComTraj used 'time_step='; some newer variants use 'dt='.
    This wrapper supports either without changing your ComTraj implementation.
    """
    try:
        traj.generate_traj(go2, gait, time_now_s, vx, vy, z, yaw_rate, dt=dt)
    except TypeError:
        traj.generate_traj(go2, gait, time_now_s, vx, vy, z, yaw_rate, time_step=dt)

# --------------------------------------------------------------------------------
# Storage Variables (CONTROL-rate logs)
# --------------------------------------------------------------------------------
x_vec = np.zeros((12, CTRL_STEPS))
mpc_force_world = np.zeros((12, CTRL_STEPS))

tau_raw = np.zeros((12, CTRL_STEPS))
tau_cmd = np.zeros((12, CTRL_STEPS))

time_log_ctrl_s = np.zeros(CTRL_STEPS)
tau_log_ctrl_Nm = np.zeros((CTRL_STEPS, 12))

# We'll log Pin-format q (xyzw) for replay
q_pin_log_ctrl = np.zeros((CTRL_STEPS, 19))

@dataclass
class FootTraj:
    pos_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    pos_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_des: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))
    vel_now: np.ndarray = field(default_factory=lambda: np.zeros((12, CTRL_STEPS)))

foot_traj = FootTraj()

mpc_update_time_ms = []
mpc_solve_time_ms = []
X_opt = None
U_opt = None

# Render-rate replay logs
time_log_render = []
q_pin_log_render = []
tau_log_render = []
next_render_t = 0.0

# --------------------------------------------------------------------------------
# Simulation Initialization (NEW APIs)
# --------------------------------------------------------------------------------

# NOTE: supply your MJCF path here
GO2_MJCF = "/home/sid/projects25/src/Go2_mujoco/unitree_go2/go2.xml"

go2 = PinGo2ModelMJCF(GO2_MJCF)
sim = MujocoGo2Model()  # scene.xml inside your class defaults
sim.model.opt.timestep = SIM_DT

viewer = sim.launch_viewer()

leg_controller = LegController()
gait = Gait(GAIT_HZ, GAIT_DUTY)
traj = ComTraj(go2)

# Initialize robot configuration (Pin format: [p, quat_xyzw, joints])
q_init = go2.state.get_q().copy()
q_init[0] = INITIAL_X_POS
q_init[1] = INITIAL_Y_POS

sim.set_state_from_pin(q_init)

# Ensure MuJoCo derived quantities ready before first sync/solve
import mujoco as mj
mj.mj_forward(sim.model, sim.data)

# Sync Pinocchio state from MuJoCo once (so go2 state matches sim exactly)
sim.update_pin_from_mu(go2)

# Initialize MPC
_call_generate_traj(
    traj, go2, gait,
    time_now_s=0.0,
    vx=x_vel_des_body,
    vy=y_vel_des_body,
    z=z_pos_des_body,
    yaw_rate=yaw_rate_des_body,
    dt=MPC_DT
)
mpc = CentroidalMPC(go2, traj)
U_opt = np.zeros((12, traj.N), dtype=float)

# --------------------------------------------------------------------------------
# Simulation Loop (same structure as your old script)
# --------------------------------------------------------------------------------
print(f"Running simulation for {RUN_SIM_LENGTH_S}s")
sim_start_time = time.perf_counter()

ctrl_i = 0
tau_hold = np.zeros(12, dtype=float)

for k in range(SIM_STEPS):
    time_now_s = float(sim.data.time)

    # --- Control update at CTRL_HZ (decimated) ---
    if (k % CTRL_DECIM) == 0 and ctrl_i < CTRL_STEPS:
        # Commands
        x_vel_des_body, y_vel_des_body, z_pos_des_body, yaw_rate_des_body = get_body_cmd(time_now_s)

        # Update Pinocchio from current MuJoCo state
        sim.update_pin_from_mu(go2)

        # Centroidal state (your new model should expose this; if it's named differently in your code,
        # change ONLY this line.)
        x_vec[:, ctrl_i] = go2.get_centroidal_state().reshape(-1)

        # Logs
        time_log_ctrl_s[ctrl_i] = time_now_s
        q_pin_log_ctrl[ctrl_i, :] = go2.state.get_q()
        # tau_log_ctrl_Nm filled after we compute tau_hold

        # Update MPC if needed
        if (ctrl_i % STEPS_PER_MPC) == 0:
            print(f"\rSimulation Time: {time_now_s:.3f} s", end="", flush=True)

            _call_generate_traj(
                traj, go2, gait,
                time_now_s=time_now_s,
                vx=x_vel_des_body,
                vy=y_vel_des_body,
                z=z_pos_des_body,
                yaw_rate=yaw_rate_des_body,
                dt=MPC_DT
            )

            sol = mpc.solve_QP(go2, traj, verbose=False)
            mpc_solve_time_ms.append(mpc.solve_time)
            mpc_update_time_ms.append(mpc.update_time)

            N = traj.N
            w_opt = sol["x"].full().flatten()
            X_opt = w_opt[: 12 * N].reshape((12, N), order="F")
            U_opt = w_opt[12 * N :].reshape((12, N), order="F")

        # Extract first GRF from MPC
        mpc_force_world[:, ctrl_i] = U_opt[:, 0]

        # Compute joint torques
        FL = leg_controller.compute_leg_torque("FL", go2, gait, mpc_force_world[LEG_SLICE["FL"], ctrl_i], time_now_s)
        tau_raw[LEG_SLICE["FL"], ctrl_i] = FL.tau
        foot_traj.pos_des[LEG_SLICE["FL"], ctrl_i] = FL.pos_des
        foot_traj.pos_now[LEG_SLICE["FL"], ctrl_i] = FL.pos_now
        foot_traj.vel_des[LEG_SLICE["FL"], ctrl_i] = FL.vel_des
        foot_traj.vel_now[LEG_SLICE["FL"], ctrl_i] = FL.vel_now

        FR = leg_controller.compute_leg_torque("FR", go2, gait, mpc_force_world[LEG_SLICE["FR"], ctrl_i], time_now_s)
        tau_raw[LEG_SLICE["FR"], ctrl_i] = FR.tau
        foot_traj.pos_des[LEG_SLICE["FR"], ctrl_i] = FR.pos_des
        foot_traj.pos_now[LEG_SLICE["FR"], ctrl_i] = FR.pos_now
        foot_traj.vel_des[LEG_SLICE["FR"], ctrl_i] = FR.vel_des
        foot_traj.vel_now[LEG_SLICE["FR"], ctrl_i] = FR.vel_now

        RL = leg_controller.compute_leg_torque("RL", go2, gait, mpc_force_world[LEG_SLICE["RL"], ctrl_i], time_now_s)
        tau_raw[LEG_SLICE["RL"], ctrl_i] = RL.tau
        foot_traj.pos_des[LEG_SLICE["RL"], ctrl_i] = RL.pos_des
        foot_traj.pos_now[LEG_SLICE["RL"], ctrl_i] = RL.pos_now
        foot_traj.vel_des[LEG_SLICE["RL"], ctrl_i] = RL.vel_des
        foot_traj.vel_now[LEG_SLICE["RL"], ctrl_i] = RL.vel_now

        RR = leg_controller.compute_leg_torque("RR", go2, gait, mpc_force_world[LEG_SLICE["RR"], ctrl_i], time_now_s)
        tau_raw[LEG_SLICE["RR"], ctrl_i] = RR.tau
        foot_traj.pos_des[LEG_SLICE["RR"], ctrl_i] = RR.pos_des
        foot_traj.pos_now[LEG_SLICE["RR"], ctrl_i] = RR.pos_now
        foot_traj.vel_des[LEG_SLICE["RR"], ctrl_i] = RR.vel_des
        foot_traj.vel_now[LEG_SLICE["RR"], ctrl_i] = RR.vel_now

        # Saturate + hold
        tau_cmd[:, ctrl_i] = np.clip(tau_raw[:, ctrl_i], -TAU_LIM, TAU_LIM)
        tau_hold = tau_cmd[:, ctrl_i].copy()
        tau_log_ctrl_Nm[ctrl_i, :] = tau_hold

        ctrl_i += 1

    # Apply held torques at every SIM step (NEW API)
    sim.set_joint_torques(tau_hold)
    sim.step()

    # Render-rate logging (Pin q for clean replay)
    t_after = float(sim.data.time)
    if t_after + 1e-12 >= next_render_t:
        time_log_render.append(t_after)
        q_pin_log_render.append(go2.state.get_q().copy())
        tau_log_render.append(tau_hold.copy())
        next_render_t += RENDER_DT

    viewer.sync()

sim_end_time = time.perf_counter()
print(
    f"\nSimulation ended."
    f"\nElapsed time: {sim_end_time - sim_start_time:.3f}s"
    f"\nControl ticks: {ctrl_i}/{CTRL_STEPS}"
)

# --------------------------------------------------------------------------------
# Plots (control-rate)
# --------------------------------------------------------------------------------
t_vec = np.arange(ctrl_i) * CTRL_DT
plot_swing_foot_traj(t_vec, foot_traj, False)
plot_mpc_result(t_vec, mpc_force_world, tau_cmd, x_vec, block=False)
plot_solve_time(mpc_solve_time_ms, mpc_update_time_ms, MPC_DT, MPC_HZ, block=True)

# --------------------------------------------------------------------------------
# Simple replay (since your new MujocoGo2Model doesn't provide replay_simulation)
# --------------------------------------------------------------------------------
time_log_render = np.asarray(time_log_render, dtype=float)
q_pin_log_render = np.asarray(q_pin_log_render, dtype=float)
tau_log_render = np.asarray(tau_log_render, dtype=float)

print("Replaying logged trajectory... close the viewer window to end replay.")

# Start a fresh passive viewer for replay
viewer_replay = sim.launch_viewer()

# replay pacing
t0_wall = time.perf_counter()
t0_sim = time_log_render[0] if len(time_log_render) > 0 else 0.0

for i in range(len(time_log_render)):
    if not viewer_replay.is_running():
        break

    # set state directly from Pin q (handles quat ordering internally)
    sim.set_state_from_pin(q_pin_log_render[i])
    sim.set_joint_torques(tau_log_render[i])

    # keep display smooth
    viewer_replay.sync()

    # realtime pacing (optional)
    if i + 1 < len(time_log_render):
        dt_sim = (time_log_render[i + 1] - time_log_render[i]) / max(REALTIME_FACTOR, 1e-9)
        if dt_sim > 0:
            time.sleep(dt_sim)

hold_until_all_fig_closed()

# Optional: full trajectory plots if your plot_helper expects X_opt/traj_ref etc.
# (kept from your old script; enable if needed)
# x0_col = x_vec[:, 0:1]
# traj_ref = np.hstack([x0_col, traj.compute_x_ref_vec()])
# traj_act = np.hstack([x0_col, X_opt])
# plot_full_traj(traj_act, traj_ref, block=True)
