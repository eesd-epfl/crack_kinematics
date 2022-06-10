#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:46:40 2021

This script contains the codes to execute the examples published in the
article "Determining crack kinematics from imaged crack patterns" by Pantoja-Rosero et. al

@author: pantoja
"""
import sys
sys.path.insert(1, '../src/')
import os
from tools_crack_kinematic import compute_kinematic_mean, load_GT_kinematic, error_kinematic_batch, plot_ablation, plot_pdf, plot_hypers_error
from batch_kinematics import run_kinematics_batch
import numpy as np


###########TOY EXAMPLES -- Real Edge##########################################
data_path = "../data/toy_examples_re_50/"

#Full edges approach
number_of_hyper_combinations = 1*50 #one per patch if 50.  samples number = 100
#Hypperparameters
#np.random.seed(27)
list_k_n_normal_feature = (5*np.ones(number_of_hyper_combinations)).astype('int')
list_omega = np.zeros(number_of_hyper_combinations)
list_crack_kinematic_full_edge = run_kinematics_batch(data_path, list_k_neighboors=None, list_m=None, list_l=None, list_k_n_normal_feature=list_k_n_normal_feature, list_omega=list_omega, full_edges=True, monte_carlo=True, normals = False, samples_number=1)
error_kinematic_batch(data_path, list_k_neighboors=None, list_m=None, list_l=None, list_k_n_normal_feature=list_k_n_normal_feature, list_omega=list_omega, full_edges=True, monte_carlo=True, read_hyper_from_results=True)
#Plotting
plot_ablation(data_path, k_n_normal_feature=None, omega=None, full_edges=True)
plot_pdf(data_path, full_edges=True)