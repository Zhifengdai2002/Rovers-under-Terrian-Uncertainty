import numpy as np
from typing import Tuple, Dict
from config import Gains, RoverParams, SimConfig
from utils import wrap_to_pi, sat
from environment import reference_trajectory, reference_curvature

class TrackingController:
    def __init__(self, gains: Gains, params: RoverParams):
        self.g = gains
        self.p = params
        self.int_v = 0.0

    def reset(self):
        self.int_v = 0.0

    def compute(
        self,
        t: float,
        state: np.ndarray,
        cfg: SimConfig
    ) -> Tuple[float, float, Dict[str, float]]:
        x, y, th, v, w = state
        xr, yr, thr, vr = reference_trajectory(t, cfg)

        e_y = -np.sin(thr)*(x - xr) + np.cos(thr)*(y - yr)
        e_th = wrap_to_pi(thr - th)

        kappa_ref = reference_curvature(xr, cfg)
        w_ff = vr * kappa_ref
        w_ref = w_ff + self.g.k_y * e_y + self.g.k_th * e_th

        e_v = vr - v
        self.int_v += e_v * cfg.dt
        F_cmd = self.g.k_vp * e_v + self.g.k_vi * self.int_v

        e_w = w_ref - w
        M_cmd = self.g.k_wp * e_w

        r = self.p.r_w
        b = self.p.b
        tau_sum = F_cmd * r
        tau_diff = (2.0 * r / b) * M_cmd

        tau_R = 0.5*(tau_sum + tau_diff)
        tau_L = 0.5*(tau_sum - tau_diff)

        tau_L_sat = sat(tau_L, self.p.tau_max)
        tau_R_sat = sat(tau_R, self.p.tau_max)

        debug = {
            "xr": xr, "yr": yr, "thr": thr, "vr": vr,
            "e_y": float(e_y), "e_th": float(e_th), "e_v": float(e_v),
            "w_ref": float(w_ref),
            "tau_L_cmd": float(tau_L), "tau_R_cmd": float(tau_R),
            "tau_L": float(tau_L_sat), "tau_R": float(tau_R_sat),
            "vr": float(vr),
            "kappa_ref": float(kappa_ref),
            "w_ff": float(w_ff)
        }
        return tau_L_sat, tau_R_sat, debug