# Robust Path Optimization for Differential-Drive Rover under Terrain Uncertainty

This project implements a physics-based simulation and optimization framework for a differential-drive rover under uncertain terrain conditions. The rover tracks a reference path using a two-layer controller, while terrain slope and rolling resistance are randomized to evaluate robustness. Controller gains are tuned using Monte Carlo evaluation and SLSQP-based constrained optimization.

## Main Features

- Differential-drive rover simulation
- Terrain uncertainty modeling (slope and rolling resistance)
- Two-layer tracking controller
- Monte Carlo robustness evaluation
- Constrained gain optimization using SLSQP
- Plotting of trajectories, errors, and wheel torques

## Optimization Objective

The controller gains are optimized to minimize robust position tracking error:

\[
\min \text{rmse\_pos\_p90}
\]

subject to tracking and actuator saturation constraints.

## How to Run

Install dependencies:

```bash
pip install numpy scipy matplotlib
