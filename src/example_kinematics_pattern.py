import os
from tools_crack_kinematic import compute_kinematic_mean 
from least_square_crack_kinematics import *
from utils_crack_kinematics import get_tn_tt

"""
This script contains examples of how to run the algorithm. Here, given a binary image
and its container folder path, the kinematic is computed for the full mask.
If desired it can be computed afterwards the kinematic values tn,tt
"""

#Example 1
data_path = "../data/real_patterns/"
mask_name = "mask_p3_000.png"
#Full edges approach
crack_kinematic_full_edge_real0 = kinematics_full_edges(data_path, mask_name, k_n_normal_feature=10, omega = 0, edges=False, normals=False, make_plots_global = True)
_ = compute_kinematic_mean(data_path, mask_name, [10,0], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)
#Finite edges approach
crack_kinematic_finite_edge_real0  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1, l=1, k_n_normal_feature=10, omega=0, edges=False, normals = False, make_plots_local=True, make_plots_global = False)
_ = compute_kinematic_mean(data_path, mask_name, [50,1,1,10,0], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True)
#Finite edges approach with bigger/extended edge1
crack_kinematic_finite_edge_real0  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=2, l=5, k_n_normal_feature=10, omega=0, edges=False, normals = False, make_plots_local=True, make_plots_global = False)
_ = compute_kinematic_mean(data_path, mask_name, [50,2,5,10,0], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True)

#Example 2
data_path = "../data/real_patterns/"
mask_name = "mask_p3_001.png"
#Finite segments approch
crack_kinematic_finite_edge_real1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.5)
_ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)
#Reading kinematics in given point
get_tn_tt(data_path, mask_name, "finite_edges", [1.5,1.5,4,10,0.], mmpx=.43)
