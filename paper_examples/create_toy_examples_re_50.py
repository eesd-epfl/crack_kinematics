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
import os
from skimage import measure
from tools_crack_kinematic import cropp_resize_images

def create_toy_re(data_path, H_params, mask_name, contour_id, folder = "original_mask", toy_name='toy_example0.png'):
    """
    Function to create toy examples used in the paper
    """
    
    #Read crack image
    #Read crack mask
    crack = cv2.imread(data_path + "/" +folder+ "/"+mask_name)
    crack = crack[:,:,0]>0
    crack = crack*255
        
    #Get contour0
    edge0, len_contour = get_edge0(crack, contour_id=contour_id)
    if len_contour != 2:
        print("WARNING: " + mask_name + " may have problems to generate toy example " + toy_name)
    edge0 = np.flip(edge0, 1)
    #Plot edge0
    plt.figure()
    plt.imshow(crack, 'gray')
    plt.plot(edge0[:,0],edge0[:,1],'r.')
    
    #Create H from parameters given
    theta = H_params[0]
    tx = H_params[1]
    ty = H_params[2]
    H = np.array([[np.cos(theta), -np.sin(theta), tx],
         [np.sin(theta), np.cos(theta), ty],
         [0,0,1]])
    
    imm = np.zeros_like(crack)
    imm = imm.astype('uint8')

    #Finding local coordinates. Place orignin in mean of edge0 coordinates
    mean_edge0 = np.mean(edge0, axis=0)
    edge0 = edge0 - mean_edge0

    #Finding edg1 as transformation of edge0
    edge1 = np.concatenate((np.copy(edge0), np.ones((len(edge0),1))), axis=1).T
    edge1 = H @ edge1 
    edge1  /= edge1[2]
    edge1  = edge1[:2].T
    
    #Taking edges to global coordinates    
    edge0 = edge0 + mean_edge0
    edge1 = edge1 + mean_edge0
    
    
    #Defining vertices of image crack
    crack_vertices = np.concatenate((edge0, np.flip(edge1,0))).astype('int')

    #save arrays
    np.save('../data/toy_examples_re_50/'+toy_name[:-4]+'_edge0', edge0) 
    np.save('../data/toy_examples_re_50/'+toy_name[:-4]+'_edge1', edge1)    
    np.save('../data/toy_examples_re_50/'+toy_name[:-4]+'_crack_vert', crack_vertices)
    np.save('../data/toy_examples_re_50/'+toy_name[:-4]+'_H_params', H_params)            
    
    #Drawing polygon
    cv2.fillPoly(imm, [crack_vertices], (255, 255, 255))

    #Saving toy example
    cv2.imwrite('../data/toy_examples_re_50/' + toy_name, imm) #cropping the image for convinience
    print("SAVED: " + mask_name + "AS " +toy_name )
    

def get_edge0(crack, contour_id=0):
    
    #Getting contour points of masks through measure of skimage library
    crack_contour = measure.find_contours(crack,100)
    
    #Select edge0 according id
    edge0 = crack_contour[contour_id]
    
    #Number of contours detected
    len_contour = len(crack_contour)
    
    return edge0, len_contour

#Input folder
data_path = "../data/toy_examples_re_50/"

#Reading real masks
#list_masks = [mask_name for mask_name in os.listdir(data_path + "original_mask/") if mask_name.endswith(".png")]
#list_masks = list_masks[:100]
#list_masks.sort()
#cropp_resize_images(data_path, list_masks)    

#Read cropped and resized
list_masks = [mask_name for mask_name in os.listdir(data_path + "c_r/") if mask_name.endswith(".png")]  
list_masks.sort()

np.random.seed(27)
theta = np.random.uniform(0,np.pi/64,100)
tx = np.random.randint(5,10,100)
ty = -np.random.randint(5,10,100)
for i, mask_name in enumerate(list_masks):
    H_params = np.array([theta[i],tx[i],ty[i]])
    toy_name = 'toy_example{}_re.png'.format(i)
    contour_id = 0
    create_toy_re(data_path, H_params, mask_name, contour_id, folder = "c_r", toy_name=toy_name)
