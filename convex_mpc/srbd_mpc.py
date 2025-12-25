import casadi as ca
import numpy as np
import scipy.sparse as sp
import time

from robot_data import PinGo2ModelMJCF
from com_traj import ComTraj

# --------------------------------------------------------------------------------
# Model Predictive Control Setting
# --------------------------------------------------------------------------------

COST_MATRIX_Q = np.diag([1, 1, 50,  10, 20, 1,  2, 2, 1,  1, 1, 1])  # (12x12)
COST_MATRIX_R = np.diag([1e-5] * 12)                                 # (12x12)

MU = 0.8
NX = 12
NU = 12

OPTS = {
    "warm_start_primal": True,
    "warm_start_dual": True,
    "osqp": {
        "eps_abs": 1e-4,
        "eps_rel": 1e-4,
        "max_iter": 1000,
        "polish": False,
        "verbose": False,
        "adaptive_rho": True,
        "check_termination": 10,
        "adaptive_rho_interval": 25,
        "scaling": 5,
        "scaled_termination": True,
    }
}

SOLVER_NAME: str = "osqp"


class CentroidalMPC:
    """
    Centroidal force MPC:
      x_k   ∈ R^12  = [p_com, rpy, v_com_world, omega_body]
      u_k   ∈ R^12  = [f_FL(3), f_FR(3), f_RL(3), f_RR(3)]  (WORLD frame)
    """

    def __init__(self, go2: PinGo2ModelMJCF, traj: ComTraj):
        self.go2 = go2

        self.Q = COST_MATRIX_Q
        self.R = COST_MATRIX_R

        self.N = traj.N  # one gait period
        self.nvars = self.N * NX + self.N * NU
        self.solve_time: float = 0.0
        self.update_time: float = 0.0

        # 1) Constant helper matrices
        self.I_block = ca.DM.eye(self.N * NX)
        ones_N_minus_1 = np.ones(self.N - 1)
        S_scipy = sp.kron(sp.diags([ones_N_minus_1], [-1]), sp.eye(NX))
        self.S_block = self._scipy_to_casadi(S_scipy)

        # 2) Constant friction constraint matrix
        self.A_ineq_static = self._precompute_friction_matrix()

        # 3) CasADi function to build block-diag dynamics pieces
        self.dyn_builder = self._create_dynamics_function()

        # 4) Initialize solver w/ correct sparsity
        self._build_sparse_matrix(traj, verbose=True)

        # Warm start memory
        self.x_prev = None
        self.lam_x_prev = None
        self.lam_a_prev = None

    # ------------------------------------------------------------
    def solve_QP(self, go2: PinGo2ModelMJCF, traj: ComTraj, verbose: bool = False):
        """
        Solve the centroidal force QP for the current iteration.
        """
        t0 = time.perf_counter()

        g, A, lba, uba = self._update_sparse_matrix(traj)
        lbx, ubx = self._compute_bounds(go2, traj)   # <-- FIXED

        t1 = time.perf_counter()

        qp_args = {
            "h": self.H_const,
            "g": g,
            "a": A,
            "lba": lba,
            "uba": uba,
            "lbx": lbx,
            "ubx": ubx,
        }

        # Warm start
        if self.x_prev is not None:
            qp_args["x0"] = self.x_prev
            if self.lam_x_prev is not None:
                qp_args["lam_x0"] = self.lam_x_prev
            if self.lam_a_prev is not None:
                qp_args["lam_a0"] = self.lam_a_prev

        sol = self.solver(**qp_args)
        t2 = time.perf_counter()

        self.update_time = (t1 - t0) * 1e3
        self.solve_time = (t2 - t1) * 1e3

        self.x_prev = sol["x"]
        self.lam_x_prev = sol.get("lam_x", None)
        self.lam_a_prev = sol.get("lam_a", None)

        if verbose:
            stats = self.solver.stats()
            print(f"[QP] update: {self.update_time:.3f} ms | solve: {self.solve_time:.3f} ms")
            print(f"[QP] status: {stats.get('return_status')}")

        return sol

    # ------------------------------------------------------------
    def _compute_bounds(self, go2: PinGo2ModelMJCF, traj: ComTraj):
        """
        Box bounds on decision variables:
        - swing leg forces = 0
        - stance leg fz >= fz_min(k) based on mg and #stance legs
        """
        N = traj.N
        nvars = self.nvars
        start_u = N * NX

        lbx_np = np.full((nvars, 1), -np.inf, dtype=float)
        ubx_np = np.full((nvars, 1),  np.inf, dtype=float)

        # Indices for forces: (12, N)
        force_block = (np.arange(NU)[:, None] + NU * np.arange(N)[None, :])
        force_idx = start_u + force_block

        contact = np.asarray(traj.contact_table, dtype=int)  # (4, N), 1=stance
        if contact.shape == (N, 4):
            contact = contact.T
        if contact.shape != (4, N):
            raise ValueError(f"traj.contact_table must be shape (4,N); got {contact.shape}")

        # ---- 1) Swing forces = 0
        leg_rows = np.array([[0, 1, 2],
                             [3, 4, 5],
                             [6, 7, 8],
                             [9, 10, 11]])

        swing = (contact == 0)
        swing_xyz = np.repeat(swing[:, None, :], 3, axis=1)  # (4, 3, N)

        mask_12N = np.zeros((12, N), dtype=bool)
        mask_12N[leg_rows.reshape(-1), :] = swing_xyz.reshape(12, N)

        swing_idx = force_idx[mask_12N]
        lbx_np[swing_idx, 0] = 0.0
        ubx_np[swing_idx, 0] = 0.0

        # ---- 2) Stance Fz bounds based on mg
        # Prefer traj.mass (from ComTraj), fallback to Pinocchio model mass
        mass = float(getattr(traj, "mass", 0.0))
        if mass <= 0.0:
            mass = float(go2.data.Ig.mass)

        mg = mass * 9.81

        fz_rows = np.array([2, 5, 8, 11])  # FL,FR,RL,RR fz
        stance_counts = np.maximum(1, np.sum(contact, axis=0))  # (N,)

        # IMPORTANT: This is a *minimum support* constraint.
        # If too large, you’ll overconstrain. If too small, robot will go ballistic.
        # Start with 0.15–0.30 and tune.
        min_support_ratio = 0.25
        fz_max = 2.5 * mg

        for k in range(N):
            fz_min_k = min_support_ratio * mg / stance_counts[k]
            for leg in range(4):
                idx = int(force_idx[fz_rows[leg], k])
                if contact[leg, k] == 1:
                    lbx_np[idx, 0] = max(lbx_np[idx, 0], fz_min_k)
                    ubx_np[idx, 0] = min(ubx_np[idx, 0], fz_max)
                else:
                    # already forced to 0 by swing constraint above, but keep safe
                    lbx_np[idx, 0] = 0.0
                    ubx_np[idx, 0] = 0.0

        return ca.DM(lbx_np), ca.DM(ubx_np)

    # ------------------------------------------------------------
    def _build_sparse_matrix(self, traj: ComTraj, verbose: bool = False):
        # 1) Hessian H (constant)
        rows, cols, vals = [], [], []

        for k in range(self.N):
            base = k * NX
            for i in range(NX):
                w = self.Q[i, i]
                if w != 0:
                    rows.append(base + i)
                    cols.append(base + i)
                    vals.append(2 * w)

        for k in range(self.N):
            base = self.N * NX + k * NU
            for i in range(NU):
                w = self.R[i, i]
                if w != 0:
                    rows.append(base + i)
                    cols.append(base + i)
                    vals.append(2 * w)

        self.H_const = ca.DM.triplet(rows, cols, ca.DM(vals), self.nvars, self.nvars)
        self.H_sp = self.H_const.sparsity()

        # 2) A sparsity from a representative assembly
        Ad_dm = ca.DM(traj.Ad)
        Bd_seq_dm = ca.DM(traj.Bd.reshape(self.N * NX, NU))
        A_init = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)
        self.A_sp = A_init.sparsity()

        # 3) Create solver
        qp = {"h": self.H_sp, "a": self.A_sp}
        self.solver = ca.conic("S", SOLVER_NAME, qp, OPTS)

        if verbose:
            print("\n[QP Init] ===== MPC QP Structure =====")
            print(f"  vars: {self.nvars} | horizon N = {self.N}")
            print("[QP Init] ✓ Initialization complete.\n")

    # ------------------------------------------------------------
    def _update_sparse_matrix(self, traj: ComTraj):
        Ad_dm = ca.DM(traj.Ad)
        Bd_seq_dm = ca.DM(traj.Bd.reshape(self.N * NX, NU))

        A_dm = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)

        # Linear cost g
        x_ref = ca.DM(traj.compute_x_ref_vec())  # (12,N)
        gx = -2 * ca.vec(ca.DM(self.Q) @ x_ref)
        g = ca.vertcat(gx, ca.DM.zeros(self.N * NU, 1))

        # Equality bounds
        if getattr(traj, "initial_x", None) is None:
            raise RuntimeError("traj.initial_x is missing; set it in ComTraj.generate_traj().")

        x0 = ca.DM(traj.initial_x)  # (12,)
        gd = ca.DM(traj.gd)         # (12,1)

        beq_first = Ad_dm @ x0 + gd
        beq_rest = ca.repmat(gd, self.N - 1, 1)
        beq = ca.vertcat(beq_first, beq_rest)

        # Inequality bounds (friction): enforce only during stance by ub=0 else ub=+inf
        n_ineq = 4 * 4 * self.N
        lb_ineq = -ca.inf * ca.DM.ones(n_ineq, 1)

        ub_np = np.inf * np.ones(n_ineq)
        ct = np.asarray(traj.contact_table, dtype=int)
        if ct.shape == (self.N, 4):
            ct = ct.T

        idx = 0
        for k in range(self.N):
            for leg in range(4):
                if ct[leg, k] == 1:
                    ub_np[idx:idx + 4] = 0.0
                idx += 4
        ub_ineq = ca.DM(ub_np)

        lb = ca.vertcat(beq, lb_ineq)
        ub = ca.vertcat(beq, ub_ineq)

        return g, A_dm, lb, ub

    # ------------------------------------------------------------
    def _assemble_A_matrix(self, Ad, Bd_seq):
        big_minus_Ad, big_minus_Bd = self.dyn_builder(Ad, Bd_seq)
        term_Ad = self.S_block @ big_minus_Ad
        A_eq = ca.horzcat(self.I_block + term_Ad, big_minus_Bd)
        return ca.vertcat(A_eq, self.A_ineq_static)

    # ------------------------------------------------------------
    def _create_dynamics_function(self):
        Ad_sym = ca.SX.sym("Ad", NX, NX)
        Bd_seq_sym = ca.SX.sym("Bd_seq", self.N * NX, NU)

        list_Ad = [-Ad_sym] * self.N
        list_Bd = []

        for k in range(self.N):
            s = k * NX
            e = (k + 1) * NX
            list_Bd.append(-Bd_seq_sym[s:e, :])

        big_Ad = ca.diagcat(*list_Ad)
        big_Bd = ca.diagcat(*list_Bd)
        return ca.Function("dyn_builder", [Ad_sym, Bd_seq_sym], [big_Ad, big_Bd])

    # ------------------------------------------------------------
    def _precompute_friction_matrix(self):
        rows, cols, vals = [], [], []
        baseU = self.N * NX
        r0 = 0

        for k in range(self.N):
            uk0 = baseU + k * NU
            for leg in range(4):
                fx, fy, fz = 3 * leg, 3 * leg + 1, 3 * leg + 2

                rows += [r0, r0]; cols += [uk0 + fx, uk0 + fz]; vals += [1.0, -MU]; r0 += 1
                rows += [r0, r0]; cols += [uk0 + fx, uk0 + fz]; vals += [-1.0, -MU]; r0 += 1
                rows += [r0, r0]; cols += [uk0 + fy, uk0 + fz]; vals += [1.0, -MU]; r0 += 1
                rows += [r0, r0]; cols += [uk0 + fy, uk0 + fz]; vals += [-1.0, -MU]; r0 += 1

        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(r0, self.nvars))
        return self._scipy_to_casadi(A_sp)

    # ------------------------------------------------------------
    @staticmethod
    def _scipy_to_casadi(M):
        M = M.tocsc()
        return ca.DM(ca.Sparsity(M.shape[0], M.shape[1], M.indptr, M.indices), M.data)
