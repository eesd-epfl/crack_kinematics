#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:09:19 2021

@author: pantoja
"""

from least_square_crack_kinematics import find_crack_kinematics, kinematics_full_edges
import os

def run_kinematics_batch(data_path, list_k_neighboors=None, list_m=None, list_l=None, list_k_n_normal_feature=None, list_omega=None, full_edges=False, ignore_global = True, monte_carlo=False, normals=False, samples_number=None, pareto=False, list_eta=None):
    """
    This function runs kinematics for a batch of crack patterns.

    Args:
        data_path (str): Data folder path
        list_k_neighboors (list, optional): List with different values of k neighbors aplied to the batch. Defaults to None.
        list_m (list, optional): List with different values of m aplied to the batch. Defaults to None.
        list_l (list, optional): List with different values of l  aplied to the batch. Defaults to None.
        list_k_n_normal_feature (liest, optional): List with different values of kn neighbors aplied to the batch. Defaults to None.
        list_omega (list, optional): List with different values of omega neighbors aplied to the batch. Defaults to None.
        full_edges (bool, optional): True if full edges approach is used. Defaults to False.
        ignore_global (bool, optional): True to just compute kinematics with local coordinates. Defaults to True.
        monte_carlo (bool, optional): True if the batch was used for montecarlo simulation. Defaults to False.
        normals (bool, optional): True to considerate normal features of edges in the residual function. Defaults to False.
        samples_number (int, optional): numper of skeleton points where kinematic is computed. Defaults to None.
        pareto (bool, optional): True if pareto is used to find optimal H. Defaults to False.
        list_eta (list, optional): List with different values of eta  aplied to the batch. Defaults to None.
    Returns:
        list_crack_kinematic (list): list of dictionaries with crack kinematics results
    """

    #Run kinematics for full or finite edges approach of all the images inside the folder data_path with format .png        
    #reading .png images that contains the crack patterns    
    list_mask_name = [mask_name for mask_name in os.listdir(data_path) if mask_name.endswith(".png")]
    list_mask_name.sort()
    list_crack_kinematic=[]
    if monte_carlo:
        #In the simulation is given for each patch an aleatory combination of hyperparameters
        #If full edges are used it is not necessart sampling points.
        #if finite edges used, we saple "samples_number" points from the skeleton. In each point we assign a different combination of hyperparameters
        if full_edges:
            #Running for all examples
            mask_id = 0
            c = 0
            for omega, k_n_normal_feature in zip(list_omega, list_k_n_normal_feature):    
                mask_name = list_mask_name[mask_id]
                c += 1
                if c%samples_number==0:
                    mask_id+=1
                list_crack_kinematic.append(kinematics_full_edges(data_path, mask_name, k_n_normal_feature=k_n_normal_feature, omega = omega, edges=True, normals=normals, make_plots_global = True))
        else:
            mask_id = 0
            c = 0
            if list_eta is None:
                for k_neighboors, m, l, omega, k_n_normal_feature in zip(list_k_neighboors, list_m, list_l, list_omega, list_k_n_normal_feature):
                    mask_name = list_mask_name[mask_id]
                    c += 1
                    if c%samples_number==0:
                        mask_id+=1
                    if l >= k_neighboors*m:
                        continue
                    list_crack_kinematic.append(find_crack_kinematics(data_path, mask_name, k_neighboors=k_neighboors, m=m, l=l, k_n_normal_feature=k_n_normal_feature, omega=omega, edges=True, normals = normals, make_plots_local=True, make_plots_global = False, ignore_global=ignore_global, monte_carlo=monte_carlo, pareto = pareto))
            else:
                for eta, m, l, omega, k_n_normal_feature in zip(list_eta, list_m, list_l, list_omega, list_k_n_normal_feature):
                    mask_name = list_mask_name[mask_id]
                    c += 1
                    if c%samples_number==0:
                        mask_id+=1
                    list_crack_kinematic.append(find_crack_kinematics(data_path, mask_name, k_neighboors=None, m=m, l=l, k_n_normal_feature=k_n_normal_feature, omega=omega, edges=True, normals = normals, make_plots_local=True, make_plots_global = False, ignore_global=ignore_global, monte_carlo=monte_carlo, pareto = pareto, eta=eta))
    else:
            
        if full_edges:
            #Running for all examples
            for mask_name in list_mask_name:    
                for omega in list_omega:
                    check_omega = 0        
                    for k_n_normal_feature in list_k_n_normal_feature:
                        #if omega==0, it just need one iteration as normals would not matter
                        if check_omega==1:
                            continue            
                        list_crack_kinematic.append(kinematics_full_edges(data_path, mask_name, k_n_normal_feature=k_n_normal_feature, omega = omega, edges=True, normals=normals, make_plots_global = True))
                        if omega==0:
                            check_omega+=1
        else:
            for mask_name in list_mask_name:
                for k_neighboors in list_k_neighboors:
                    for m in list_m:
                        #Only make sense iterate in l if m>1. If m==1 the just compute once
                        check_m = 0
                        for l in list_l:
                            #Only make sense if m>1 (edge1 bigger than edge0) and kn*m>l
                            if check_m == len(list_omega)*len(list_k_n_normal_feature):
                                continue
                            if l >= k_neighboors*m:
                                continue
                            for omega in list_omega:
                                check_omega = 0        
                                for k_n_normal_feature in list_k_n_normal_feature:
                                    #if omega==0, it just need one iteration as normals would not matter
                                    if check_omega==1:
                                        continue            
                                    #list_crack_kinematic.append(find_crack_kinematic3(data_path, mask_name, k_neighboors=k_neighboors, m=m, l=l, k_n_normal_feature=k_n_normal_feature, omega=omega, edges=True, normals = normals, make_plots_local=True, make_plots_global = False, ignore_global=ignore_global, pareto=pareto))
                                    list_crack_kinematic.append(find_crack_kinematics(data_path, mask_name, k_neighboors=k_neighboors, m=m, l=l, k_n_normal_feature=k_n_normal_feature, omega=omega, edges=True, normals = normals, make_plots_local=True, make_plots_global = False, ignore_global=ignore_global, pareto=pareto))
                                    if omega==0:
                                        check_omega+=1
                                    if m==1:
                                        check_m+=1

    return list_crack_kinematic    