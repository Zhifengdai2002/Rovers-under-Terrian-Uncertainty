import numpy as np
from config import RoverParams, EnvParams, SimConfig
from utils import save_results, load_results
from simulation import simulate_once
from optimization import find_feasible_baseline, optimize_gains
from plotting import (plot_episode, plot_summary, plot_terrain_height,
                      plot_tracking_error_comparison, plot_optimization_history)

def main():
    MODE = "run"   # "run" or "plot"
    params = RoverParams()
    env = EnvParams()
    results_file = "saved_rover_results.pkl"
    cfg = SimConfig(
        dt=0.02, T=30.0,
        rmse_xy_max=1.4, rmse_heading_max=0.2, sat_frac_max=0.4,
        v_ref=1.2, path_type="s_curve",
    )

    if MODE == "plot":
        data = load_results(results_file)
        print("Loaded saved results. Re-plotting only...")
        plot_summary(data["metrics_base"], data["metrics_opt"])
        plot_terrain_height(data["out_base"], "Representative Terrain Height Profile")
        plot_tracking_error_comparison(data["out_base"], data["out_opt"])
        plot_episode(data["out_base"], "Baseline Episode", params)
        plot_episode(data["out_opt"], "Optimized Episode", params)
        plot_optimization_history(data["history"])
        return

    g0, metrics_base = find_feasible_baseline(params, env, cfg, n_candidates=30, n_trials_screen=20, seed=123)

    print("\n=== Baseline evaluation ===")
    for k, v in metrics_base.items(): print(f"{k:>14s}: {v:.4f}")

    print("\n=== Optimizing gains ===")
    x0 = np.array([g0.k_y, g0.k_th, g0.k_vp, g0.k_vi, g0.k_wp], dtype=float)
    g_opt, metrics_opt, history = optimize_gains(x0, params, env, cfg)

    print("\n=== Optimized gains ===")
    print(g_opt)
    
    print("\n=== Plot one episode (baseline vs optimized) ")
    out_base = simulate_once(g0, params, env, cfg, seed=42, randomize_env=True)
    out_opt  = simulate_once(g_opt, params, env, cfg, seed=42, randomize_env=True)

    save_results(results_file, {
        "g0": g0, "g_opt": g_opt,
        "metrics_base": metrics_base, "metrics_opt": metrics_opt,
        "out_base": out_base, "out_opt": out_opt, "history": history
    })
    print(f"Results saved to {results_file}")

    plot_summary(metrics_base, metrics_opt)
    plot_terrain_height(out_base, "Representative Terrain Height Profile")
    plot_tracking_error_comparison(out_base, out_opt)
    plot_episode(out_base, "Baseline Episode", params)
    plot_episode(out_opt, "Optimized Episode", params)
    plot_optimization_history(history)

if __name__ == "__main__":
    main()