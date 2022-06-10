#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:46:40 2021
Script with for monte carlo simulations shown in the article
"Determining crack kinematics from imaged crack patterns" by Pantoja-Rosero et. al
@author: pantoja
"""
import sys
sys.path.insert(1, '../src/')
import os
from tools_crack_kinematic import compute_kinematic_mean, load_GT_kinematic, error_kinematic_batch, plot_ablation, plot_pdf, plot_hypers_error,plot_hypers_time,plot_pdf_time
from batch_kinematics import run_kinematics_batch
import numpy as np


###########TOY EXAMPLES -- Real Edge##########################################
#Finite edges approach
data_path = "../data/toy_examples_re_50_pareto/"
#Hypperparameters
samples_number = 100
number_of_hyper_combinations = samples_number*50 # samples_patch * No. patches
np.random.seed(27)
list_k_neighboors = np.random.randint(30,251,number_of_hyper_combinations)
list_m = np.random.randint(10,31,number_of_hyper_combinations)/10
list_l = np.ones(number_of_hyper_combinations).astype('int') #This will not change with pareto
list_k_n_normal_feature = np.ones(number_of_hyper_combinations)
list_omega = np.zeros(number_of_hyper_combinations)

#Running kinematics on images located in data_path under defined hyperparameters
list_crack_kinematic_finite_edge = run_kinematics_batch(data_path, list_k_neighboors=list_k_neighboors, list_m=list_m, list_l=list_l, list_k_n_normal_feature=list_k_n_normal_feature, list_omega=list_omega, full_edges=False, monte_carlo=True, samples_number=samples_number, normals=False, ignore_global=True, pareto=True)
#computer batch error. absolute error for 3dof and 2dof. 2dof given also as unique
error_kinematic_batch(data_path, list_k_neighboors=list_k_neighboors, list_m=list_m, list_l=list_l, list_k_n_normal_feature=list_k_n_normal_feature, list_omega=list_omega, full_edges=False, monte_carlo=True, read_hyper_from_results=True)
#Plotting
plot_pdf(data_path, full_edges=False)
plot_hypers_error(data_path, full_edges=False)#, pareto=True)
plot_hypers_time(data_path, full_edges=False)
plot_pdf_time(data_path, full_edges=False)

                 