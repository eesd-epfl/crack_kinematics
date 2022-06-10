"""
Created on Sun Sep 19 15:46:40 2021

This script contains the codes to create the examples published in the
article "Determining crack kinematics from imaged crack patterns" by Pantoja-Rosero et. al

@author: pantoja
"""
import sys
sys.path.insert(1, '../src/')
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2 
from skimage import measure


def generate_noisy_examples(noise_intensity, edge0, edge1, x_noise_edge0, y_noise_edge0, x_noise_edge1, y_noise_edge1, input_path, output_path, toy_name):
    """
    Function to create toy examples used in the paper
    """
    mask = cv2.imread(input_path + toy_name + '.png')    
    H_gt = np.load(input_path + toy_name + '_H_params.npy')
    
    for i in noise_intensity:
        #edge0
        edge0_noisy = np.copy(edge0)
        edge0_noisy[:,0] += i*x_noise_edge0
        edge0_noisy[:,1] += i*y_noise_edge0
        #edge1
        edge1_noisy = np.copy(edge1)
        edge1_noisy[:,0] += i*x_noise_edge1
        edge1_noisy[:,1] += i*y_noise_edge1
        np.save(output_path + toy_name + '_noise_{}_edge0.npy'.format(i), edge0_noisy)
        np.save(output_path + toy_name + '_noise_{}_edge1.npy'.format(i), edge1_noisy)
        np.save(output_path + toy_name + '_edge0_noise_{}.npy'.format(i), edge0_noisy)
        np.save(output_path + toy_name + '_edge1_noise_{}.npy'.format(i), edge1_noisy)
        
        #save mask with the same name of the contours
        cv2.imwrite(output_path + toy_name + '_noise_{}.png'.format(i) , mask)
        #save GT with the same name of the contours
        np.save(output_path + toy_name + '_noise_{}_H_params.npy'.format(i), H_gt)
    
    #read and plot
    for i in noise_intensity:
        edge0_noisy = np.load(output_path + toy_name + '_edge0_noise_{}.npy'.format(i))
        edge1_noisy = np.load(output_path + toy_name + '_edge1_noise_{}.npy'.format(i))
        plt.plot(edge0_noisy[:,0], edge0_noisy[:,1], '.')
        plt.plot(edge1_noisy[:,0], edge1_noisy[:,1], '.')
        plt.axis('equal')
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '.pdf')
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '.png')
        plt.close()
    
    #read and plot
    for i in noise_intensity:
        edge0_noisy = np.load(output_path + toy_name + '_edge0_noise_{}.npy'.format(i))
        edge1_noisy = np.load(output_path + toy_name + '_edge1_noise_{}.npy'.format(i))
        plt.plot(edge0_noisy[:,0], edge0_noisy[:,1], '.')
        plt.plot(edge1_noisy[:,0], edge1_noisy[:,1], '.')
        plt.axis('equal')
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '_noise_{}.pdf'.format(i))
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '_noise_{}.png'.format(i))
        plt.close()
    
    
    for i in noise_intensity:
        
        plt.imshow(mask)
        edge0_noisy = np.load(output_path + toy_name + '_edge0_noise_{}.npy'.format(i))
        edge1_noisy = np.load(output_path + toy_name + '_edge1_noise_{}.npy'.format(i))
        plt.plot(edge0_noisy[:,0], edge0_noisy[:,1], '.')
        plt.plot(edge1_noisy[:,0], edge1_noisy[:,1], '.')
        plt.axis('equal')
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '_mask.pdf')
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '_mask.png')
        plt.close()
    
    for i in noise_intensity:
        
        plt.imshow(mask)
        edge0_noisy = np.load(output_path + toy_name + '_edge0_noise_{}.npy'.format(i))
        edge1_noisy = np.load(output_path + toy_name + '_edge1_noise_{}.npy'.format(i))
        plt.plot(edge0_noisy[:,0], edge0_noisy[:,1],c = (.0,.5,1.), marker = '.')
        plt.plot(edge1_noisy[:,0], edge1_noisy[:,1],c = (1.,.2,0.), marker = '.')
        plt.axis('equal')
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '_mask_noise_{}.pdf'.format(i))
        plt.savefig(output_path + '/noisy_edges_' + toy_name + '_mask_noise_{}.png'.format(i))
        plt.close()
    

#User interaction
input_path = '../data/toy_examples_re/'
output_path = '../data/toy_examples_re_noise/'

#The noise is applied to all the points of the edges in 0,1,3,5,10,15,20,30,50,100%
noise_intensity = [0.,.01,.03,.05,.1,.15,.2,.3,.5,1.]

#Toy 0#######################
toy_name = 'toy_example0_re'
edge0_toy_example0 = np.load(input_path + toy_name + '_edge0.npy')
edge1_toy_example0 = np.load(input_path + toy_name + '_edge1.npy')
#Generating noise 
np.random.seed(27)
x_noise_toy_example0_edge0, y_noise_toy_example0_edge0 = np.random.normal(0,1,len(edge0_toy_example0)) , np.random.normal(0,1,len(edge0_toy_example0))
x_noise_toy_example0_edge1, y_noise_toy_example0_edge1 = np.random.normal(0,1,len(edge1_toy_example0)) , np.random.normal(0,1,len(edge1_toy_example0))
#Generating noisy edges
generate_noisy_examples(noise_intensity, edge0_toy_example0, edge1_toy_example0, x_noise_toy_example0_edge0, y_noise_toy_example0_edge0, x_noise_toy_example0_edge1, y_noise_toy_example0_edge1, input_path, output_path, toy_name)


#Toy 1#######################
toy_name = 'toy_example1_re'
edge0_toy_example1 = np.load(input_path + toy_name + '_edge0.npy')
edge1_toy_example1 = np.load(input_path + toy_name + '_edge1.npy')
#Generating noise 
#np.random.seed(27)
x_noise_toy_example1_edge0, y_noise_toy_example1_edge0 = np.random.normal(0,1,len(edge0_toy_example1)) , np.random.normal(0,1,len(edge0_toy_example1))
x_noise_toy_example1_edge1, y_noise_toy_example1_edge1 = np.random.normal(0,1,len(edge1_toy_example1)) , np.random.normal(0,1,len(edge1_toy_example1))
#Generating noisy edges
generate_noisy_examples(noise_intensity, edge0_toy_example1, edge1_toy_example1, x_noise_toy_example1_edge0, y_noise_toy_example1_edge0, x_noise_toy_example1_edge1, y_noise_toy_example1_edge1, input_path, output_path, toy_name)


#Toy 2#######################
toy_name = 'toy_example2_re'
edge0_toy_example2 = np.load(input_path + toy_name + '_edge0.npy')
edge1_toy_example2 = np.load(input_path + toy_name + '_edge1.npy')
#Generating noise 
#np.random.seed(27)
x_noise_toy_example2_edge0, y_noise_toy_example2_edge0 = np.random.normal(0,1,len(edge0_toy_example2)) , np.random.normal(0,1,len(edge0_toy_example2))
x_noise_toy_example2_edge1, y_noise_toy_example2_edge1 = np.random.normal(0,1,len(edge1_toy_example2)) , np.random.normal(0,1,len(edge1_toy_example2))
#Generating noisy edges
generate_noisy_examples(noise_intensity, edge0_toy_example2, edge1_toy_example2, x_noise_toy_example2_edge0, y_noise_toy_example2_edge0, x_noise_toy_example2_edge1, y_noise_toy_example2_edge1, input_path, output_path, toy_name)
