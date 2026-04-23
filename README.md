# Robust Path Optimization for Differential-Drive Rover under Terrain Uncertainty

This project develops a physics-based simulation and optimization framework for a differential-drive rover operating under uncertain terrain conditions. The rover tracks a reference trajectory using a two-layer controller, while terrain slope and rolling resistance are randomized to evaluate robustness. Controller gains are tuned through Monte Carlo evaluation and constrained nonlinear optimization using SLSQP.

## Overview

The main goal of this project is to improve rover trajectory tracking performance under terrain uncertainty while respecting actuator limits. The framework combines:

- rover dynamics simulation
- terrain uncertainty modeling
- closed-loop tracking control
- Monte Carlo robustness evaluation
- constrained controller gain optimization

## Main Components

### Rover Model
The rover is modeled as a differential-drive system with position, heading, linear velocity, and angular velocity states. The simulation includes:

- translational and rotational dynamics
- slope-induced gravitational effects
- rolling resistance
- wheel torque saturation

### Terrain Uncertainty
Terrain uncertainty is introduced through randomized:

- rolling resistance coefficient
- slope scaling factor

This allows the rover to be tested under multiple terrain realizations.

### Controller Structure
A two-layer controller is used:

- **Outer loop:** computes desired yaw rate from lateral and heading errors
- **Inner loop:** regulates speed and yaw rate, then maps commands to wheel torques

### Optimization
The controller gains are optimized to minimize robust position tracking error under uncertainty. The objective is based on Monte Carlo evaluation, and constraints are imposed on:

- position tracking RMSE
- heading RMSE
- actuator saturation fraction

The optimization is solved using **SLSQP**.

## Workflow

1. Define rover, terrain, and simulation parameters  
2. Randomly sample candidate controller gains for baseline search  
3. Evaluate candidate gains using Monte Carlo simulation  
4. Select a feasible baseline controller  
5. Run SLSQP optimization from the baseline solution  
6. Compare baseline and optimized performance  

## Outputs

The project produces:

- baseline and optimized controller gains
- statistical tracking metrics
- trajectory comparison plots
- tracking error plots
- wheel torque plots

## Example Figures

You can place result images in a `figures/` folder and display them here.

```markdown
![Baseline vs Optimized Trajectory](figures/trajectory_comparison.png)
![Tracking Error](figures/tracking_error.png)

## How to Run

Install dependencies:

```bash
pip install numpy scipy matplotlib
