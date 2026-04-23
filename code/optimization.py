import numpy as np
from typing import Tuple, Dict
from scipy.optimize import minimize
from config import Gains, RoverParams, EnvParams, SimConfig
from utils import clamp
from simulation import evaluate_gains

def gains_from_vector(x: np.ndarray) -> Gains:
    def pos(z, lo=1e-4, hi=1000.0):
        return float(clamp(z, lo, hi))
    return Gains(k_y=pos(x[0]), k_th=pos(x[1]), k_vp=pos(x[2]), k_vi=pos(x[3]), k_wp=pos(x[4]))

def objective_for_optimizer(x, params, env, cfg, n_trials=10, seed0=0) -> float:
    g = gains_from_vector(x)
    metrics = evaluate_gains(g, params, env, cfg, n_trials=n_trials, seed0=seed0)
    mu_pos = metrics["rmse_pos_mean"]
    std_pos = metrics["rmse_pos_std"]
    k = 1.28  
    return float(mu_pos + k * std_pos)

def optimize_gains(
    x0: np.ndarray,
    params: RoverParams,
    env: EnvParams,
    cfg: SimConfig
) -> Tuple[Gains, Dict[str, float], Dict[str, list]]:
    history = {"iter": [], "obj": [], "k_y": [], "k_th": [], "k_vp": [], "k_vi": [], "k_wp": []}

    def callback(xk):
        g = gains_from_vector(xk)
        obj = objective_for_optimizer(xk, params, env, cfg, n_trials=10, seed0=0)
        history["iter"].append(len(history["iter"]))
        history["obj"].append(obj)
        history["k_y"].append(g.k_y)
        history["k_th"].append(g.k_th)
        history["k_vp"].append(g.k_vp)
        history["k_vi"].append(g.k_vi)
        history["k_wp"].append(g.k_wp)

    def cons_rmse_pos(x):
        m = evaluate_gains(gains_from_vector(x), params, env, cfg, n_trials=10, seed0=0)
        return cfg.rmse_xy_max - m["rmse_pos_p90"]

    def cons_rmse_heading(x):
        m = evaluate_gains(gains_from_vector(x), params, env, cfg, n_trials=10, seed0=0)
        return cfg.rmse_heading_max - m["rmse_th_p90"]

    def cons_sat_frac(x):
        m = evaluate_gains(gains_from_vector(x), params, env, cfg, n_trials=10, seed0=0)
        return cfg.sat_frac_max - m["sat_frac_mean"]

    cons = [
        {"type": "ineq", "fun": cons_rmse_pos},
        {"type": "ineq", "fun": cons_rmse_heading},
        {"type": "ineq", "fun": cons_sat_frac},
    ]

    bounds = [(0.01, 8.0), (0.2, 30.0), (10.0, 300.0), (0.0, 300.0), (1.0, 120.0)]

    res = minimize(
        objective_for_optimizer, x0=x0, args=(params, env, cfg, 6, 0),
        method="SLSQP", bounds=bounds, constraints=cons,
        callback=callback, options={"maxiter": 60, "disp": True}
    )

    best = gains_from_vector(res.x)
    metrics = evaluate_gains(best, params, env, cfg, n_trials=12, seed0=100)
    metrics["opt_success"] = bool(res.success)
    metrics["opt_message"] = str(res.message)
    metrics["x_opt"] = res.x.tolist()

    return best, metrics, history

def _sample_gain_vector(rng: np.random.Generator) -> np.ndarray:
    k_y  = rng.uniform(0.2, 8.0)
    k_th = rng.uniform(0.2, 12.0)
    k_vp = rng.uniform(10.0, 300.0)
    k_vi = rng.uniform(0.0, 200.0)
    k_wp = rng.uniform(1.0, 120.0)
    return np.array([k_y, k_th, k_vp, k_vi, k_wp], dtype=float)

def _constraint_violation(metrics: Dict[str, float], cfg: SimConfig) -> float:
    v1 = max(0.0, (metrics["rmse_pos_p90"] - cfg.rmse_xy_max) / max(cfg.rmse_xy_max, 1e-9))
    v2 = max(0.0, (metrics["rmse_th_p90"]  - cfg.rmse_heading_max) / max(cfg.rmse_heading_max, 1e-9))
    v3 = max(0.0, (metrics["sat_frac_mean"] - cfg.sat_frac_max) / max(cfg.sat_frac_max, 1e-9))
    return float(v1 + v2 + v3)

def find_feasible_baseline(
    params: RoverParams, env: EnvParams, cfg: SimConfig,
    n_candidates: int = 250, n_trials_screen: int = 6,
    seed: int = 123, verbose: bool = True
) -> Tuple[Gains, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    best_feasible = None
    best_feasible_score = float("inf")  
    best_any = None
    best_any_violation = float("inf")

    if verbose: print(f"\n=== Feasible baseline search (Candidates={n_candidates}) ===")

    for i in range(n_candidates):
        x = _sample_gain_vector(rng)
        g = gains_from_vector(x)
        metrics = evaluate_gains(g, params, env, cfg, n_trials=n_trials_screen, seed0=seed + 1000 + i)

        feasible = (
            (metrics["rmse_pos_p90"] <= cfg.rmse_xy_max) and
            (metrics["rmse_th_p90"]  <= cfg.rmse_heading_max) and
            (metrics["sat_frac_mean"] <= cfg.sat_frac_max)
        )

        if feasible:
            current_score = metrics["rmse_pos_mean"] + 1.28 * metrics["rmse_pos_std"]
            if current_score < best_feasible_score:
                best_feasible_score = current_score
                best_feasible = g

        vio = _constraint_violation(metrics, cfg)
        if vio < best_any_violation:
            best_any_violation = vio
            best_any = g

        if verbose and (i+1) % max(1, n_candidates//10) == 0:
            msg = f"[{i+1:>4d}/{n_candidates}] "
            msg += f"best feasible score={best_feasible_score:.3f}" if best_feasible else f"best violation={best_any_violation:.3f}"
            print(msg)

    if best_feasible is not None:
        if verbose: print("\nFound feasible baseline")
        return best_feasible, evaluate_gains(best_feasible, params, env, cfg, n_trials=20, seed0=seed + 5000)

    if verbose: print("\nNo fully feasible baseline found. Returning least-violating candidate.")
    return best_any, evaluate_gains(best_any, params, env, cfg, n_trials=20, seed0=seed + 5000)