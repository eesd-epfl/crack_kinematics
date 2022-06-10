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
from tools_crack_kinematic import compute_kinematic_mean, error_kinematic_batch, plot_pdf
from batch_kinematics import run_kinematics_batch
from least_square_crack_kinematics import *


#The examples here are the presented in the paper
###########EXAMPLES -- Real Edge Introduction###############################################
data_path = "../data/real_patterns/"
mask_name = "mask_p3_000.png"
#Full edges##
#No normals
crack_kinematic_full_edge_real0 = kinematics_full_edges(data_path, mask_name, k_n_normal_feature=10, omega = 0, edges=False, normals=False, make_plots_global = True)
_ = compute_kinematic_mean(data_path, mask_name, [10,0], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)

#Finite edges
#no normals
crack_kinematic_finite_edge_real0  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1, l=1, k_n_normal_feature=10, omega=0, edges=False, normals = False, make_plots_local=True, make_plots_global = False)
_ = compute_kinematic_mean(data_path, mask_name, [50,1,1,10,0], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True)

#Finite edges - extended E1
#no normals
crack_kinematic_finite_edge_real0  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=2, l=5, k_n_normal_feature=10, omega=0, edges=False, normals = False, make_plots_local=True, make_plots_global = False)
_ = compute_kinematic_mean(data_path, mask_name, [50,2,5,10,0], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True)

##########TOY EXAMPLES -- LINES###############################################
data_path = "../data/toy_examples_lines/"
#Full edges approach
#Hypperparameters
list_k_n_normal_feature = [3,]
list_omega = [0,]
#Running kinematics on images located in data_path under defined hyperparameters
list_crack_kinematic_full_edge_toy_lines = run_kinematics_batch(data_path, list_k_neighboors=None, list_m=None, list_l=None, list_k_n_normal_feature=list_k_n_normal_feature, list_omega=list_omega, full_edges=True)
#computer batch error. absolute error for 3dof and 2dof. 2dof given also as unique
error_kinematic_batch(data_path, list_k_neighboors=None, list_m=None, list_l=None, list_k_n_normal_feature=list_k_n_normal_feature, list_omega=list_omega, full_edges=True)

###########TOY EXAMPLES -- Real Edge###############################################
data_path = "../data/toy_examples_re/"


mask_name = "toy_example0_re.png"
#Full edges##
#No normals
crack_kinematic_full_edge_toy_real0 = kinematics_full_edges(data_path, mask_name, k_n_normal_feature=10, omega = 0., edges=True, normals=False, make_plots_local=True, make_plots_global = True)
_ = compute_kinematic_mean(data_path, mask_name, [10,0.], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)

#finite segments
#No normals
crack_kinematic_finite_edge_toy_real0  = find_crack_kinematics(data_path, mask_name, k_neighboors=200, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=True, normals = False, make_plots_local=True, make_plots_global = False, ignore_global=True, eta=1.5)
_ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)
mask_name = "toy_example1_re.png"
#Full edges##
#No normals
crack_kinematic_full_edge_toy_real1 = kinematics_full_edges(data_path, mask_name, k_n_normal_feature=10, omega = 0., edges=True, normals=False, make_plots_local=True, make_plots_global = True)
_ = compute_kinematic_mean(data_path, mask_name, [10,0.], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)

#finite segments
#No normals
crack_kinematic_finite_edge_toy_real1  = find_crack_kinematics(data_path, mask_name, k_neighboors=200, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=True, normals = False, make_plots_local=True, make_plots_global = False, ignore_global=True, eta=1.5)
_ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "toy_example2_re.png"
#Full edges##
#No normals
crack_kinematic_full_edge_toy_real2 = kinematics_full_edges(data_path, mask_name, k_n_normal_feature=10, omega = 0., edges=True, normals=False, make_plots_local=True, make_plots_global = True)
_ = compute_kinematic_mean(data_path, mask_name, [10,0.], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)

#finite segments
#No normals
crack_kinematic_finite_edge_toy_real2  = find_crack_kinematics(data_path, mask_name, k_neighboors=200, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=True, normals = False, make_plots_local=True, make_plots_global = False, ignore_global=True, eta=1.5)
_ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

###########TOY EXAMPLES -- Real Edge -- Noise addition###############################################
data_path = "../data/toy_examples_re_noise/"

#The noise is applied to all the points of the edges in 0,1,3,5,10,15,20,30,50,100%
noise_intensity = [0.0,.01,.03,.05,.1,.15,.2,.3,.5,1.0]
toy_name_list = ['toy_example0_re','toy_example1_re','toy_example2_re']

#Full edges##
for toy_name in toy_name_list:    
    for i in noise_intensity:    
        mask_name = toy_name + '_noise_{}.png'.format(i)        
        #No normals
        crack_kinematic_full_edge_toy_real0 = kinematics_full_edges(data_path, mask_name, k_n_normal_feature=10, omega = 0, edges=True, normals=False, make_plots_local=True, make_plots_global = True)
        _ = compute_kinematic_mean(data_path, mask_name, [10,0], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)
        
#Compute errors and plotting
#Full edges##
for toy_name in toy_name_list:    
    for i in noise_intensity:  
        mask_name = toy_name + '_noise_{}.png'.format(i)        
        #No normals
        _ = compute_kinematic_mean(data_path, mask_name, [10,0], crack_kinematic=None, full_edges=True, local_trans = False, abs_disp = False)
       
error_kinematic_batch(data_path, full_edges=True, read_hyper_from_results = True, monte_carlo=True) #not montecarlo but the function works like montecarlo
plot_pdf(data_path, full_edges=True)


#finite segments eta
for toy_name in toy_name_list:    
    for i in noise_intensity:        
        mask_name = toy_name + '_noise_{}.png'.format(i)
        #No normals
        crack_kinematic_finite_edge_toy_real0  = find_crack_kinematics(data_path, mask_name, k_neighboors=200, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=True, normals = False, make_plots_local=True, make_plots_global = False, ignore_global=True, eta = 1.5)
        _ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)
                    

#Compute errors and plotting
error_kinematic_batch(data_path, full_edges=False, read_hyper_from_results = True, monte_carlo=True, list_eta = []) #not montecarlo but the function works like montecarlo
plot_pdf(data_path, full_edges=False, use_eta=False)

###########EXAMPLES -- Real Patterns###############################################
data_path = "../data/real_patterns/"
mask_name = "mask_p3_000.png"

#finite segments
#No normals
crack_kinematic_finite_edge_real0  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.5)
_ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "mask_p3_001.png"
#finite segments
#No normals
crack_kinematic_finite_edge_real1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.5)
_ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "mask_p3_002.png"
#finite segments
#No normals
crack_kinematic_finite_edge_real2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1.5, l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.5)
_ = compute_kinematic_mean(data_path, mask_name, [1.5,1.5,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

###########EXAMPLES -- DIC pattern initial paper version###############################################


data_path = "../data/DIC/Pantoja-Rosero_2021/"
mask_name = "SVS_2_LS43_to_LS44_RS6_0000_0_mask.png" #Too slow
#finite segments
#No normals
crack_kinematic_dic  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)
#PLOT
batch_name = data_path.split('/')[-2]
dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(1., 1., 4,10,0.)
with open(dir_save+'crack_kinematic.json', 'r') as fp:
    crack_kinematic_dic = json.load(fp)
plot_n_t_kinematic(data_path, mask_name, crack_kinematic_dic,  dir_save=dir_save,plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True,local_trans = True, abs_disp = True, dot_size=1, cmap_='jet')

data_path = "../data/DIC/Pantoja-Rosero_2021/PR_2021_patches_0/"    
    
mask_name = "dic_patch0_y3806_x2854.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch1_y2012_x2779.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch2_y3723_x601.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_3  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

data_path = "../data/DIC/Pantoja-Rosero_2021/PR_2021_patches_0_v2/"    
    
mask_name = "dic_patch0_y2396_x1965.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch1_y2118_x1746.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch2_y3211_x2523.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_3  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


data_path = "../data/DIC/Pantoja-Rosero_2021/PR_2021_patches_1_v2/"    
    
mask_name = "dic_patch0_y2062_x1698.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch1_y3120_x2432.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet')
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

data_path = "../data/DIC/Pantoja-Rosero_2021/PR_2021_patches_1/"    
    
mask_name = "dic_patch0_y3633_x541.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_1_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch1_y4010_x2606.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_1_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch2_y2759_x2109.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_1_3  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch3_y1620_x1212.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_1_4  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "dic_patch4_y1522_x2531.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_1_5  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

###########EXAMPLES -- DIC pattern last paper version####

data_path = "../data/DIC/Pantoja-Rosero_2021/PR_2021_patches_v2/"    
    
mask_name = "dic_patch0_y2062_x1698.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet', resolution=0.43)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)
#PLOT
batch_name = data_path.split('/')[-2]
dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(1., 1., 4,10,0.)
with open(dir_save+'crack_kinematic.json', 'r') as fp:
    crack_kinematic_dic = json.load(fp)
plot_n_t_kinematic(data_path, mask_name, crack_kinematic_dic,  dir_save=dir_save,plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True,local_trans = True, abs_disp = True, cmap_='jet', resolution=0.43)

mask_name = "dic_patch1_y3120_x2432.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet', resolution=0.43)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)
#PLOT
batch_name = data_path.split('/')[-2]
dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(1., 1., 4,10,0.)
with open(dir_save+'crack_kinematic.json', 'r') as fp:
    crack_kinematic_dic = json.load(fp)
plot_n_t_kinematic(data_path, mask_name, crack_kinematic_dic,  dir_save=dir_save,plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True,local_trans = True, abs_disp = True, cmap_='jet', resolution=0.43)

mask_name = "dic_patch0_y3723_x601.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_4  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1., cmap_='jet', resolution=0.43)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)
#PLOT
batch_name = data_path.split('/')[-2]
dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(1., 1., 4,10,0.)
with open(dir_save+'crack_kinematic.json', 'r') as fp:
    crack_kinematic_dic = json.load(fp)
plot_n_t_kinematic(data_path, mask_name, crack_kinematic_dic,  dir_save=dir_save,plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True,local_trans = True, abs_disp = True, cmap_='jet', resolution=0.43)

###########EXAMPLES -- real Building##############################################
data_path = "../data/buildingB/"
mask_name = "DJI_0998_mask_filtered.png" #Too slow
#finite segments
#No normals
crack_kinematic_buildingB  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

#PLOT
batch_name = data_path.split('/')[-2]
dir_save = '../results/' + batch_name + '/' + mask_name[:-4] + '/' + 'finite_edges/eta{}_m{}_l{}_knnor{}_omega{}/'.format(1., 1., 4,10,0.)
#with open(dir_save+'crack_kinematic.json', 'r') as fp:
    #crack_kinematic_buildingB = json.load(fp)
plot_n_t_kinematic(data_path, mask_name, crack_kinematic_buildingB,  dir_save=dir_save,plot_trans_n=True, plot_trans_t=True, plot_trans_t_n=True,local_trans = True, abs_disp = True, dot_size=1)

### USING PATCHES
data_path = "../data/buildingB/buildingB_patches/"    
    
mask_name = "building_patch0_y1153_x2921.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "building_patch1_y364_x2015.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "building_patch2_y1583_x1304.png" 
#finite segments
#No normals
crack_kinematic_dic_patch_0_3  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

###########EXAMPLES -- Flexure beam####

data_path = "../data/beam/"    
    
mask_name = "_IMG_20220509_105555_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

mask_name = "_IMG_20220509_105611_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

###Patches - Pattern1
data_path = "../data/beam/patterns/pattern1/"    
    
mask_name = "beam_patch0_y627_x2613_mask__IMG_20220509_105555_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_0  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch1_y1551_x2127_mask__IMG_20220509_105555_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch2_y2008_x1956_mask__IMG_20220509_105555_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch3_y2717_x2004_mask__IMG_20220509_105555_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_3  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch4_y3186_x1877_mask__IMG_20220509_105555_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_4  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)

###Patches - Pattern2
data_path = "../data/beam/patterns/pattern2/"    
    
mask_name = "beam_patch0_y655_x2589_mask__IMG_20220509_105611_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_0  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch1_y1268_x2601_mask__IMG_20220509_105611_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_1  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch2_y1706_x2692_mask__IMG_20220509_105611_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_2  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch3_y2386_x2191_mask__IMG_20220509_105611_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_3  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)


mask_name = "beam_patch4_y2661_x2740_mask__IMG_20220509_105611_mask.png" 
#finite segments
#No normals
crack_kinematic_beam_1_4  = find_crack_kinematics(data_path, mask_name, k_neighboors=50, m=1., l=4, k_n_normal_feature=10, omega=0., edges=False, normals = False, make_plots_local=True, make_plots_global = False, eta = 1.)
_ = compute_kinematic_mean(data_path, mask_name, [1.,1.,4,10,0.], crack_kinematic=None, full_edges=False, local_trans = True, abs_disp = True, use_eta = True)