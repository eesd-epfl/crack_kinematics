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

def create_toy_re(H_params, mask_name, contour_id, toy_name='toy_example0.png'):
    """
    Function to create toy examples used in the paper
    """
    
    #Read crack image
    #Read crack mask
    crack = cv2.imread("../data/toy_examples_re/real/"+mask_name+".png")
    crack = crack[:,:,0]>0
    crack = crack*255
    crack = crack[10:240,10:240] #cropping the image for convinience
    
    #Saving new images
    #Get contour0
    edge0 = get_edge0(crack, contour_id=contour_id)
    edge0 = np.flip(edge0, 1)
    #Plot edge0
    plt.figure()
    plt.imshow(crack, 'gray')
    #plt.plot(edge0[:,1],edge0[:,0],'r.')
    plt.plot(edge0[:,0],edge0[:,1],'r.')
    
    #Create H from parameters given
    #H_type == "euclidean":
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
    
    plt.figure()
    plt.imshow(imm,'gray')
    plt.plot(edge0[:,0], edge0[:,1], 'bo')
    plt.plot(edge1[:,0], edge1[:,1], 'ro')

    #Taking edges to global coordinates    
    edge0 = edge0 + mean_edge0
    edge1 = edge1 + mean_edge0
    
    np.save('../data/toy_examples_re/'+toy_name[:-4]+'_edge0_no_crop', edge0) 
    np.save('../data/toy_examples_re/'+toy_name[:-4]+'_edge1_no_crop', edge1)  
    
    #Defining vertices of image crack
    crack_vertices = np.concatenate((edge0, np.flip(edge1,0))).astype('int')

    #save arrays
    #bcs of the cropping the image for convinience, -10 in x and y and delete points out of 0-210
    edge0 = edge0 - np.array([10,10])
    edge0_ = np.array([ed for ed in edge0 if 0<ed[0]<210 and 0<ed[1]<210])
    edge1 = edge1 - np.array([10,10])
    edge1_ = np.array([ed for ed in edge1 if 0<ed[0]<210 and 0<ed[1]<210])
    #crack_vertices = crack_vertices - np.array([10,10])
    crack_vertices_ = np.array([ed for ed in crack_vertices if 0<ed[0]<210 and 0<ed[1]<210])

    np.save('../data/toy_examples_re/'+toy_name[:-4]+'_edge0', edge0) 
    np.save('../data/toy_examples_re/'+toy_name[:-4]+'_edge1', edge1)    
    np.save('../data/toy_examples_re/'+toy_name[:-4]+'_crack_vert', crack_vertices)
    np.save('../data/toy_examples_re/'+toy_name[:-4]+'_H_params', H_params)            
    
    #Drawing polygon
    cv2.fillPoly(imm, [crack_vertices], (255, 255, 255))

    #Saving toy example
    cv2.imwrite('../data/toy_examples_re/' + toy_name, imm[10:220,10:220]) #cropping the image for convinience
    
    plt.figure()
    plt.imshow(imm[10:220,10:220], 'gray')
    plt.figure()
    plt.imshow(imm[10:220,10:220], 'gray')
    plt.plot(edge0_[:,0], edge0_[:,1], 'bo')
    plt.plot(edge1_[:,0], edge1_[:,1], 'ro')
    

def get_edge0(crack, contour_id=0):
    """
    Function to create toy examples used in the paper
    """
    
    #Getting contour points of masks through measure of skimage library
    crack_contour = measure.find_contours(crack,100)
    
    #Select edge0 according id
    edge0 = crack_contour[contour_id]
    
    return edge0

#User interaction

#Toy 1: simulating mode I
mask_name = "mask_toy_000"
theta = 0*np.pi/64
tx = +10
ty = +0
H_params = np.array([theta,tx,ty])
toy_name = 'toy_example0_re.png'
contour_id = 0
create_toy_re(H_params, mask_name, contour_id, toy_name=toy_name)

#Toy 2: simulating mode II 
mask_name = "mask_toy_001"
theta = 0*np.pi/64
tx = 5
ty = -5
H_params = np.array([theta,tx,ty])
toy_name = 'toy_example1_re.png'
contour_id = 0
create_toy_re(H_params, mask_name, contour_id, toy_name=toy_name)

mask_name = "mask_toy_002"
theta = 1*np.pi/64
tx = +10
ty = +10
H_params = np.array([theta,tx,ty])
toy_name = 'toy_example2_re.png'
contour_id = 0
create_toy_re(H_params, mask_name, contour_id, toy_name=toy_name)