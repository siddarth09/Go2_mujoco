import numpy as np 
from robot_data import PinGo2ModelMJCF 
from foot_step_planner import Gait 
from dataclasses import dataclass 



KP_SWING = np.diag([400,400,400])
KD_SWING = np.diag([75,75,75])

LEG_INDEX = {
    "FL":0,
    "FR":1,
    "RL":2,
    "RR":3, 
}

# Mapping from leg name to the joint torque slice in (C*dq + g)
JOINT_SLICES = {
    "FL": slice(6, 9),
    "FR": slice(9, 12),
    "RL": slice(12, 15),
    "RR": slice(15, 18),
}


@dataclass
class LegOutput:
    tau:np.ndarray 
    pos_des:np.ndarray # desired pose 
    pos_now: np.ndarray #current pose 
    vel_des: np.ndarray #desired velocity 
    vel_now: np.ndarray #current velocity 


class LegController():
    def __init__(self):
        self.last_mask = np.array([2,2,2,2])

    
    def compute_leg_torque(
            self,
            leg:str,
            go2: PinGo2ModelMJCF,
            gait: Gait,
            contact_force:np.ndarray,
            t: float
    ):
        
        # Extract Parameters 
        leg_idx = LEG_INDEX[leg]
        joint_slice= JOINT_SLICES[leg]

        q = go2.state.get_q()
        dq = go2.state.get_dq()
        foot_pos_now = go2.get_foot_position_world(leg)
        J_leg = go2.get_leg_foot_jacobian_world_3x3(q,leg)

        foot_vel_now = J_leg @ dq[joint_slice]

        M = go2.mass_matrix()
        C = go2.coriolis_matrix()
        g = go2.gravity() 

        mask = gait.compute_current_mask(t)

        foot_pos_des = foot_pos_now.copy() 
        foot_vel_des = np.zeros(3)
        tau = np.zeros(3)

        if self.last_mask[leg_idx] != mask[leg_idx] and mask [leg_idx] == 0: 

            setattr(self, f"{leg}_takeoff_time", t)
            traj, td_pos = gait.compute_swing_traj_and_touchdown(go2, leg)
            setattr(self, f"{leg}_traj", traj)
            setattr(self, f"{leg}_td_pos", td_pos)


        # SWING TIME 

        if mask[leg_idx] == 0 :
          
            traj = getattr(self, f"{leg}_traj")
            t0   = getattr(self, f"{leg}_takeoff_time")

            foot_pos_des, foot_vel_des, foot_acc_des = traj(t - t0)

            pos_err = foot_pos_des - foot_pos_now
            vel_err = foot_vel_des - foot_vel_now

            M_full = go2.mass_matrix()
            M_leg  = M_full[joint_slice, joint_slice]

            Lamda = np.linalg.inv(J_leg @ np.linalg.inv(M_leg) @ J_leg.T)

            Jdot_dq = go2.get_leg_Jdot_dq_world(q, dq, leg)

            f_ff = Lamda @ (foot_acc_des - Jdot_dq)

            force = KP_SWING @ pos_err + KD_SWING @ vel_err + f_ff

            tau = J_leg.T @ force + (C @ dq + g)[joint_slice]



        else: 
            foot_vel_des = np.zeros(3)
            tau = J_leg.T @ (contact_force) + (C @ dq + g)[joint_slice]
        
        if leg in ("FL","FR","RL","RR"):
            tau = np.clip(tau, [-23.7, -23.7, -45.43], [23.7, 23.7, 45.43])

        self.last_mask[leg_idx]= mask[leg_idx]

        return LegOutput(
            tau=tau,
            pos_des=foot_pos_des,
            pos_now=foot_pos_now,
            vel_des=foot_vel_des,
            vel_now=foot_vel_now,
        )