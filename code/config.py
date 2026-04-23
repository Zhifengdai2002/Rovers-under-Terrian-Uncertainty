from dataclasses import dataclass
from typing import Tuple

@dataclass
class RoverParams:
    m: float = 25.0          
    r_w: float = 0.15        
    b: float = 0.50          
    tau_max: float = 7.0 
    g: float = 9.81          

    J_v: float = 6.0         
    J_w: float = 2.0         
    c_v: float = 2.0         
    c_w: float = 1.2         

@dataclass
class EnvParams:
    c_rr_nom: float = 0.04
    c_rr_range: Tuple[float, float] = (0.02, 0.06)
    slope_deg_max: float = 12.0

@dataclass
class SimConfig:
    dt: float = 0.02
    T: float = 30.0
    rmse_xy_max: float = 0.25    
    rmse_heading_max: float = 1.0 
    sat_frac_max: float = 0.4   
    v_ref: float = 1.2         
    path_type: str = "s_curve"    
    energy_max: float = 1200.0   

@dataclass
class Gains:
    k_y: float = 2.0     
    k_th: float = 3.0   
    k_vp: float = 120.0 
    k_vi: float = 40.0   
    k_wp: float = 30.0