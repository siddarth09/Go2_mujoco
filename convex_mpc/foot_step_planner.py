import numpy as np 
from robot_data import PinGo2ModelMJCF

# --------------------------------------------------------------------------------
# Gait Setting
# --------------------------------------------------------------------------------

PHASE_OFFSET = np.array([0.5, 0.0, 0.0, 0.5]).reshape(4)    # trotting gait
HEIGHT_SWING = 0.1 # Height of the swing leg trajectory apex


class Gait():
    def __init__(self,freq,duty):

        self.gait_duty = duty 
        self.gait_hz = freq 
        self.gait_period = 1/freq 
        self.stance_time = self.gait_duty * self.gait_period 
        self.swing_time = (1-self.gait_duty) * self.gait_period 

    def compute_current_mask(self,time):
        """ Given the current time, it decides which legs are in 
        contact (stance) and which are in swing, for this instant."""

        mask = self.compute_contact_table(time,0,1)

        # Output : [1,0,0,1]
        return mask 
    
    def compute_contact_table(self,t0:float,dt:float,N:int)->np.ndarray:

        """It predicts the contact state of each leg over the entire MPC horizon."""

        t = t0 + np.arange(N,device="cpu")*dt 
        t = t + dt/2 

        phases = np.mod(PHASE_OFFSET[:,None]+t[None,:]/self.gait_period,1.0)

        contact_table = (phases < self.gait_duty).astype(np.int32)

        """
        contact_table =
                [[1 1 0 0 1 1]
                [0 0 1 1 0 0]
                [0 0 1 1 0 0]
                [1 1 0 0 1 1]]

        """
        return contact_table
    

    def compute_touchdown_world(self, go2: PinGo2ModelMJCF, leg: str):

        base_pos = go2.state.base_pos

        # BODY → WORLD velocity
        R_bw = go2.get_base_rotation_body_to_world()
        v_world = R_bw @ go2.state.base_lin_vel_body

        yaw = go2.get_base_rpy()[2]
        yaw_rate = go2.get_base_angular_velocity_body()[2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        R_z = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0]
        ])

        hip_offset = go2.get_hip_offset_base(leg)
        body_pos = np.array([base_pos[0], base_pos[1], 0.0])
        hip_pos_world = body_pos + R_z @ hip_offset

        T = self.swing_time + 0.5 * self.stance_time
        pred_time = T / 2.0

        pos_nominal = np.array([hip_pos_world[0], hip_pos_world[1], 0.02])

        pos_drift = np.array([
            v_world[0] * pred_time,
            v_world[1] * pred_time,
            0.0
        ])

        dtheta = yaw_rate * pred_time
        r_xy = pos_nominal[:2] - body_pos[:2]

        yaw_correction = np.array([
            -dtheta * r_xy[1],
            dtheta * r_xy[0],
            0.0
        ])

        return pos_nominal + pos_drift + yaw_correction

    
    def compute_swing_traj_and_touchdown(self, go2: PinGo2ModelMJCF, leg: str):

        base_pos = go2.state.base_pos
        pos_com = go2.get_com_position()
        vel_com = go2.get_com_velocity()

        R_bw = go2.get_base_rotation_body_to_world()
        v_world = R_bw @ go2.state.base_lin_vel_body

        yaw = go2.get_base_rpy()[2]
        yaw_rate = go2.get_base_angular_velocity_body()[2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        R_z = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0]
        ])

        hip_offset = go2.get_hip_offset_base(leg)
        foot_pos = go2.get_foot_position_world(leg)

        body_pos = np.array([base_pos[0], base_pos[1], 0.0])
        hip_pos_world = body_pos + R_z @ hip_offset

        T = self.swing_time + 0.5 * self.stance_time
        pred_time = T / 2.0

        # Gains
        k_v_x, k_p_x = 0.4 * T, 0.1
        k_v_y, k_p_y = 0.2 * T, 0.05

        pos_nominal = np.array([hip_pos_world[0], hip_pos_world[1], 0.02])

        pos_drift = np.array([
            v_world[0] * pred_time,
            v_world[1] * pred_time,
            0.0
        ])

        pos_correction = np.array([
            k_p_x * (pos_com[0]),
            k_p_y * (pos_com[1]),
            0.0
        ])

        vel_correction = np.array([
            k_v_x * (vel_com[0]),
            k_v_y * (vel_com[1]),
            0.0
        ])

        dtheta = yaw_rate * pred_time
        r_xy = pos_nominal[:2] - body_pos[:2]

        yaw_correction = np.array([
            -dtheta * r_xy[1],
            dtheta * r_xy[0],
            0.0
        ])

        pos_touchdown = (
            pos_nominal
            + pos_drift
            + pos_correction
            + vel_correction
            + yaw_correction
        )

        swing_traj = self.make_swing_trajectory(
            foot_pos,
            pos_touchdown,
            self.swing_time,
            h_sw=HEIGHT_SWING
        )

        return swing_traj, pos_touchdown

    def make_swing_trajectory(self,p0,pf,t_swing,h_sw):

        p0 = np.asarray(p0,dtype=float)
        p1 = np.asarray(pf,dtype=float)
        T = float(t_swing)
        dp = pf - p0 


        def eval_at(t):
            # phase s in [0,1]
            s = np.clip(t / T, 0.0, 1.0)

            # Minimum-jerk basis and its derivatives
            mj   = 10*s**3 - 15*s**4 + 6*s**5
            dmj  = 30*s**2 - 60*s**3 + 30*s**4           # d(mj)/ds
            d2mj = 60*s    - 180*s**2 + 120*s**3         # d^2(mj)/ds^2

            # Base (x,y,z) trajectory
            p = p0 + dp * mj
            v = (dp * dmj) / T
            a = (dp * d2mj) / (T**2)

            # Optional smooth z-bump: b(s)=64*s^3*(1-s)^3, with zero vel/acc at ends
            if h_sw != 0.0:
                b    = 64 * s**3 * (1 - s)**3
                db   = 192 * s**2 * (1 - s)**2 * (1 - 2*s)           # db/ds
                d2b  = 192 * ( 2*s*(1 - s)**2*(1 - 2*s)
                            - 2*s**2*(1 - s)*(1 - 2*s)
                            - 2*s**2*(1 - s)**2 )                  # d^2b/ds^2

                p[2] += h_sw * b
                v[2] += h_sw * db / T
                a[2] += h_sw * d2b / (T**2)

            return p, v, a

        return eval_at
    