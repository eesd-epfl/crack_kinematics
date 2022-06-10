#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 25 16:09:53 2022

This script contains the codes to plot results related with noisy examples published in the
article "Determining crack kinematics from imaged crack patterns" by Pantoja-Rosero et. al

@author: pantoja
"""
import sys
sys.path.insert(1, '../src/')
import numpy as np
import matplotlib.pyplot as plt
import json


def plot_C(data, colors_list, labels_list, noise_values, C_data=None, approach = 'full_edges', number_patterns = 3, dofs = "three", variable = "C"):
    '''
    
    Function to plot relative percent differences using predictions    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    colors_list : TYPE
        DESCRIPTION.
    labels_list : TYPE
        DESCRIPTION.
    noise_values : TYPE
        DESCRIPTION.
    approach : TYPE, optional
        DESCRIPTION. The default is 'full_edges'.
    number_patterns : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    None.

    '''
    
    #Plot name
    if variable == "C":
        p_name = "Relative"
        l_name = "Relative percent"
        p_var = "$C$ $[\%]$"
    elif variable == "D":
        p_name = "Absolute"
        l_name = "Absolute"
        p_var = "$\Delta$ $[px]$"
    
    
    if C_data is not None:
        C_data=C_data
    else:               
        if variable == "C":
            C_data = np.abs(100 * 2 * ((data[1:] - data[0]) / (data[1:] + data[0])))
        elif variable == "D":
            C_data = np.abs(data[1:] - data[0])
    
    for i in range(number_patterns):
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Gausian Noise $[\%]$", fontsize = 12)
        ax.set_ylabel(l_name + " difference " + p_var, fontsize = 12)
        
        for j in range(len(labels_list)):
            x1 = C_data[:,(len(labels_list))*i+j] * (C_data[:,(len(labels_list))*i+j]<199) #When the value is 200% in this cases is related with GT equal to zero and predictions approximately zero. Then those values are rounded to zero
            col = colors_list[j]
            labels = labels_list[j]
            
            ax.scatter(noise_values, x1, c=col, marker='.', label='Displacement '+labels, s=100)
        
        pattern = i
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xticks(fontsize=12)
        ax.legend(prop={"size":10})
        plt.tight_layout()
        plt.show()
        fig.savefig('../results/'+p_name+'_diff_pattern_{}_{}_{}_dofs.pdf'.format(pattern, approach, dofs))
        fig.savefig('../results/'+p_name+'_diff_pattern_{}_{}_{}_dofs.png'.format(pattern, approach, dofs))
        plt.close()

def plot_C_json(dofs, approach, number_patterns, variable = "C"):
    
    if variable == "C":
        v_name = "C_"
    elif variable == "D":
        v_name = ""
    
    if dofs=='two':
        key_dofs = v_name+'two_dofs'
        labels_list = ["$t'_x$", "$t'_y$"]
    else:
        key_dofs = v_name+'three_dofs'
        labels_list = ["$\\theta$", "$t_x$", "$t_y$"]
    
    #Reading dictionary with relative percent differences
    colors_list = ['#1f77b4', '#ff7f0e', '#7f7f7f', '#17becf']
    
    noise_values = np.array([0,1,3,5,10,15,20,30,50,100])
    if approach=='full_edges':
        with open('../results/toy_examples_re_noise/full_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
    else:
        with open('../results/toy_examples_re_noise/finite_edges_error_kinematic_batch.json', 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
    C_data = np.zeros(shape=[len(noise_values),len(labels_list)*number_patterns])
    ci=0
    cj=0
    for i, key in enumerate(dict_error_kinematic_batch['C_two_dofs']):
        if i%len(noise_values)==0 and i>0:
            ci=0
            if dofs=='two':
                cj+=2        
            else:
                cj+=3        
        if dofs=='two':
            Ctx = dict_error_kinematic_batch[key_dofs][key][1][0]
            Cty = dict_error_kinematic_batch[key_dofs][key][1][1]
            C_data[ci,cj] = Ctx
            C_data[ci,cj+1] = Cty
        else:
            Ctheta = dict_error_kinematic_batch[key_dofs][key][1][0]
            Ctx = dict_error_kinematic_batch[key_dofs][key][1][1]
            Cty = dict_error_kinematic_batch[key_dofs][key][1][2]
            C_data[ci,cj] = Ctheta
            C_data[ci,cj+1] = Ctx
            C_data[ci,cj+2] = Cty        
        
        ci+=1
        
    plot_C(0, colors_list, labels_list, noise_values, C_data=C_data, approach = approach, number_patterns = number_patterns, dofs = dofs, variable = variable)
    


#Registration results for edges with noise. Cols 1-3: Pattern 1, Cols 4-6: Pattern 2, Cols 7-9: Pattern 3
#Rows GT, Noise 0%, 1%,3%,5%, 10%,15%,20%,30%, 50%,100%

#With table values
#Full edges approach
data_f = np.array([[0.00,     10.00, 0.00,    0.00,    5.00, 5.00, 4.91e-2, 10.00, 10.00],
                 [7.54e-12, 10.00, 2.68e-7, 5.07e-3, 3.26, 5.01, 4.85e-2,  8.18, 11.28],
                 [4.82e-6,  10.00, 9.98e-4, 5.05e-3, 3.26, 5.01, 4.83e-2,  8.76, 10.93],
                 [1.56e-5,  10.00, 3.04e-3, 5.12e-3, 3.26, 5.01, 4.83e-2,  8.76, 10.92],
                 [2.37e-5,  10.00, 5.65e-3, 5.20e-3, 3.24, 5.01, 4.85e-2,  8.62, 11.07],
                 [8.01e-5,  10.00, 1.34e-2, 2.34e-3, 4.16, 5.00, 4.87e-2,  7.55, 11.70],
                 [2.81e-4,   9.99, 1.87e-2, 2.84e-3, 4.11, 5.00, 4.91e-2,  6.93, 12.12],
                 [4.31e-4,   9.98, 2.81e-3, 5.68e-5, 4.91, 4.97, 4.93e-2,  6.84, 12.18],
                 [4.06e-4,   9.97, 7.73e-2, 4.79e-4, 4.96, 4.96, 4.93e-2,  7.12, 11.98],
                 [3.19e-4,   9.94, 8.55e-4, 6.43e-4, 4.71, 4.95, 4.86e-2,  7.79, 11.62],
                 [1.71e-3,   9.94, 4.01e-1, 4.50e-3, 4.06, 4.87, 4.70e-2,  7.75, 11.65]])


#finite segment edges approach
data_s = np.array([[0.00,  10.00, 0.00,    0.00,     5.00, 5.00, 4.91e-2, 10.00, 10.00],
                 [5.17e-5, 10.00, 4.08e-3, 5.35e-3,  4.20, 5.03, 5.13e-2, 8.19, 11.03],
                 [6.75e-5, 10.00, 3.40e-3, 5.64e-3,  3.86, 4.99, 5.12e-2, 8.05, 11.15],
                 [1.07e-4, 10.00, 4.33e-3, 5.53e-3,  3.89, 4.99, 5.10e-2, 8.22, 11.01],
                 [1.10e-4, 10.00, 4.71e-3, 3.80e-3,  3.89, 4.92, 5.15e-2, 8.11, 11.10],
                 [1.53e-4, 10.00, 9.76e-3, 3.29e-3,  4.20, 4.95, 5.16e-2, 7.98, 11.19],
                 [3.16e-4,  9.99, 1.71e-2, 3.50e-3,  4.25, 4.95, 5.17e-2, 7.94, 11.22],
                 [4.03e-4,  9.99, 1.93e-2, 3.42e-3,  4.43, 4.99, 5.19e-2, 7.87, 11.27],
                 [6.65e-4,  9.98, 7.24e-2, 4.74e-3,  4.32, 4.98, 5.22e-2, 7.79, 11.31],
                 [1.76e-3,  9.93, 1.67e-1, 5.86e-3,  4.00, 4.95, 5.29e-2, 7.74, 11.39],
                 [3.12e-3,  9.96, 3.89e-1, 8.43e-3,  3.88, 4.96, 5.49e-2, 7.60, 11.45]])



colors_list = ['#1f77b4', '#ff7f0e', '#7f7f7f', '#17becf']
labels_list = ["$\\theta$", "$t_x$", "$t_y$"]
noise_values = np.array([0,1,3,5,10,15,20,30,50,100])
#Relative percent difference
#Full edges approach
plot_C(data_f, colors_list, labels_list, noise_values, approach = 'full_edges', number_patterns = 3)
#Finite segment edges approach
plot_C(data_s, colors_list, labels_list, noise_values, approach = 'finite_edges', number_patterns = 3)

#Relative percent difference from json file
#Using theta, tx, ty
plot_C_json("three", "full_edges", 3)
plot_C_json("three", "finite_edges", 3)
#Using tx' ty' instead of theta, tx, ty
plot_C_json("two", "full_edges", 3)
plot_C_json("two", "finite_edges", 3)

#Absolute difference from json file
#Using theta, tx, ty
plot_C_json("three", "full_edges", 3, variable = "D")
plot_C_json("three", "finite_edges", 3, variable = "D")
#Using tx' ty' instead of theta, tx, ty
plot_C_json("two", "full_edges", 3, variable = "D")
plot_C_json("two", "finite_edges", 3, variable = "D")