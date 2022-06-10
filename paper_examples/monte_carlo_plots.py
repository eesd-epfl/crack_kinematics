#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:27:07 2021

Script with plots for monte carlo simulations shown in the article
"Determining crack kinematics from imaged crack patterns" by Pantoja-Rosero et. al

@author: pantoja
"""
import os
import sys
sys.path.insert(1, '../src/')
from tools_crack_kinematic import *

#FULL PARAMS########################

#Plot pdfs in sinlge plot
plot_name = 'full_params_'
data_paths = ["../data/toy_examples_re_50_lambda/",
              "../data/toy_examples_re_50_lambda_eta/",
              "../data/toy_examples_re_50_pareto/",
              "../data/toy_examples_re_50_pareto_eta/"]
use_eta_list = [False, True, False, True]
labels_list = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4']
#colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']#, '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors_list = ['#1f77b4', '#ff7f0e', '#7f7f7f', '#17becf']
#error
plot_pdf_full(data_paths, full_edges = False, use_eta_list = use_eta_list, labels_list = labels_list, colors_list = colors_list, plot_name = plot_name)
#time
plot_pdf_time_full(data_paths, full_edges = False, use_eta_list = use_eta_list, labels_list = labels_list, colors_list = colors_list, plot_name = plot_name)


#plot hyper error in single plot
#k
plot_hypers_error_full([data_paths[0],data_paths[2]], use_eta_list=[use_eta_list[0],use_eta_list[2]], parameter = 'k', labels_list = [labels_list[0],labels_list[2]], colors_list = [colors_list[0],colors_list[2]], plot_name = plot_name)
#eta
plot_hypers_error_full([data_paths[1],data_paths[3]], use_eta_list=[use_eta_list[1],use_eta_list[3]], parameter = 'eta', labels_list = [labels_list[1],labels_list[3]], colors_list = [colors_list[1],colors_list[3]], plot_name = plot_name)
#lambda
plot_hypers_error_full([data_paths[0],data_paths[1]], use_eta_list=[use_eta_list[0],use_eta_list[1]], parameter = 'lambda', labels_list = [labels_list[0],labels_list[1]], colors_list = [colors_list[0],colors_list[1]], plot_name = plot_name)
#mu
plot_hypers_error_full(data_paths, use_eta_list=use_eta_list, parameter = 'mu', labels_list = labels_list, colors_list = colors_list, plot_name = plot_name)

#plot hyper time in single plot
#k
plot_hypers_error_full([data_paths[0],data_paths[2]], use_eta_list=[use_eta_list[0],use_eta_list[2]], parameter = 'k', labels_list = [labels_list[0],labels_list[2]], colors_list = [colors_list[0],colors_list[2]], var='time', plot_name = plot_name)
#eta
plot_hypers_error_full([data_paths[1],data_paths[3]], use_eta_list=[use_eta_list[1],use_eta_list[3]], parameter = 'eta', labels_list = [labels_list[1],labels_list[3]], colors_list = [colors_list[1],colors_list[3]], var='time', plot_name = plot_name)
#lambda
plot_hypers_error_full([data_paths[0],data_paths[1]], use_eta_list=[use_eta_list[0],use_eta_list[1]], parameter = 'lambda', labels_list = [labels_list[0],labels_list[1]], colors_list = [colors_list[0],colors_list[1]], var='time', plot_name = plot_name)
#mu
plot_hypers_error_full(data_paths, use_eta_list=use_eta_list, parameter = 'mu', labels_list = labels_list, colors_list = colors_list, var='time', plot_name = plot_name)



#LAMBDA 1 PARETO COMPARISON #######################
plot_name = 'comparison_'
data_paths = ["../data/toy_examples_re_50_lambda_comparison/",
              "../data/toy_examples_re_50_lambda_eta_comparison/",
              "../data/toy_examples_re_50_pareto_comparison/",
              "../data/toy_examples_re_50_pareto_eta_comparison/"]
use_eta_list = [False, True, False, True]
labels_list = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4']
#colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']#, '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors_list = ['#1f77b4', '#ff7f0e', '#7f7f7f', '#17becf']
#error
plot_pdf_full(data_paths, full_edges = False, use_eta_list = use_eta_list, labels_list = labels_list, colors_list = colors_list, plot_name = plot_name)
#time
plot_pdf_time_full(data_paths, full_edges = False, use_eta_list = use_eta_list, labels_list = labels_list, colors_list = colors_list, plot_name = plot_name)


#plot hyper error in single plot
#k
plot_hypers_error_full([data_paths[0],data_paths[2]], use_eta_list=[use_eta_list[0],use_eta_list[2]], parameter = 'k', labels_list = [labels_list[0],labels_list[2]], colors_list = [colors_list[0],colors_list[2]], plot_name = plot_name)
#eta
plot_hypers_error_full([data_paths[1],data_paths[3]], use_eta_list=[use_eta_list[1],use_eta_list[3]], parameter = 'eta', labels_list = [labels_list[1],labels_list[3]], colors_list = [colors_list[1],colors_list[3]], plot_name = plot_name)
#lambda
plot_hypers_error_full([data_paths[0],data_paths[1]], use_eta_list=[use_eta_list[0],use_eta_list[1]], parameter = 'lambda', labels_list = [labels_list[0],labels_list[1]], colors_list = [colors_list[0],colors_list[1]], plot_name = plot_name)
#mu
plot_hypers_error_full(data_paths, use_eta_list=use_eta_list, parameter = 'mu', labels_list = labels_list, colors_list = colors_list, plot_name = plot_name)

#plot hyper time in single plot
#k
plot_hypers_error_full([data_paths[0],data_paths[2]], use_eta_list=[use_eta_list[0],use_eta_list[2]], parameter = 'k', labels_list = [labels_list[0],labels_list[2]], colors_list = [colors_list[0],colors_list[2]], var='time', plot_name = plot_name)
#eta
plot_hypers_error_full([data_paths[1],data_paths[3]], use_eta_list=[use_eta_list[1],use_eta_list[3]], parameter = 'eta', labels_list = [labels_list[1],labels_list[3]], colors_list = [colors_list[1],colors_list[3]], var='time', plot_name = plot_name)
#lambda
plot_hypers_error_full([data_paths[0],data_paths[1]], use_eta_list=[use_eta_list[0],use_eta_list[1]], parameter = 'lambda', labels_list = [labels_list[0],labels_list[1]], colors_list = [colors_list[0],colors_list[1]], var='time', plot_name = plot_name)
#mu
plot_hypers_error_full(data_paths, use_eta_list=use_eta_list, parameter = 'mu', labels_list = labels_list, colors_list = colors_list, var='time', plot_name = plot_name)

