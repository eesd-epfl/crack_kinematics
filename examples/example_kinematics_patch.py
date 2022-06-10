#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:56:01 2022

@author: pantoja

This script contains examples of how to run the algorithm. Here, given a binary image
and its container folder path, the kinematic is computed in the closest skeleton point
to an image coordinate. The image coordinate can be given beforehand o during the process
by clicking over the image. The functions crop a window with given size in the original 
binary image around the image coordinate. Check functions for more details. 

"""
import sys
sys.path.insert(1, '../src/')
from utils_crack_kinematics import find_kinematics_patch_finite, find_kinematics_patch_full_edge


#Data path
data_path = "../data/DIC/Pantoja-Rosero_2021/"
#Full crack mask name
mask_name = "SVS_2_LS43_to_LS44_RS6_0000_0_mask.png"
#Finding kinematics for a patch by clicking center point

#If sk pt is given (in which kinematics is wanted)
#For finite edge segments
find_kinematics_patch_finite(data_path, mask_name, sk_pt = [1753,2140], size=(256,256), mmpx=.43, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
#For full edge segments (works if only a single crack is seen)
find_kinematics_patch_full_edge(data_path, mask_name, sk_pt = [1753,2140], size=(256,256), mmpx=.43, k_n_normal_feature=10, omega = 0., edges=False, normals=False, make_plots_local=True, make_plots_global = True)

#If sk pt is NOT given (in which kinematics is wanted) -- click in the mask
#For finite edge segments
find_kinematics_patch_finite(data_path, mask_name, size=(256,256), mmpx = .43, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
#For full edge segments (works if only a single crack is seen)
find_kinematics_patch_full_edge(data_path, mask_name, size=(256,256), mmpx= .43, k_n_normal_feature=10, omega = 0., edges=False, normals=False, make_plots_local=True, make_plots_global = True)
