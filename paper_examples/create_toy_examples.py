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

def create_toy(H_params, n_edge0_points, toy_name='toy_example0.png'):
    """
    Function to create toy examples used in the paper
    """    
    #Create H from parameters given
    theta = H_params[0]
    tx = H_params[1]
    ty = H_params[2]
    H = np.array([[np.cos(theta), -np.sin(theta), tx],
         [np.sin(theta), np.cos(theta), ty],
         [0,0,1]])
    
    imm = np.zeros((256,256,3), 'uint8')

    plt.figure()
    plt.imshow(imm)
    print('Please click {} points'.format(n_edge0_points))
    edge0 = np.array(pylab.ginput(n_edge0_points,200))
    print('you clicked:',edge0)

    #Finding local coordinates. Place orignin in mean of edge0 coordinates
    mean_edge0 = np.mean(edge0, axis=0)
    edge0 = edge0 - mean_edge0

    #Finding edg1 as transformation of edge0
    edge1 = np.concatenate((np.copy(edge0), np.ones((len(edge0),1))), axis=1).T
    edge1 = H @ edge1 
    edge1  /= edge1[2]
    edge1  = edge1[:2].T
    
    plt.figure()
    plt.imshow(imm)
    plt.plot(edge0[:,0], edge0[:,1], 'bo')
    plt.plot(edge1[:,0], edge1[:,1], 'ro')

    #Taking edges to global coordinates    
    edge0 = edge0 + mean_edge0
    edge1 = edge1 + mean_edge0
    
    #edges as set of points simulating detector
    edge0_ =[]
    for i in range(len(edge0)-1):
        x,y = edge0[i]
        dx,dy = (edge0[i+1]-edge0[i])/(np.linalg.norm((edge0[i+1]-edge0[i])))
        if dy>0:
            while x<edge0[i+1][0] and y<edge0[i+1][1]:
                edge0_.append([x,y])
                x+=dx
                y+=dy
        else:
            while x<edge0[i+1][0] and y>edge0[i+1][1]:
                edge0_.append([x,y])
                x+=dx
                y+=dy
                
    edge0_ = np.array(edge0_)
    edge1_ =[]
    for i in range(len(edge1)-1):
        x,y = edge1[i]
        dx,dy = (edge1[i+1]-edge1[i])/(np.linalg.norm((edge1[i+1]-edge1[i])))
        if dy>0:
            while x<edge1[i+1][0] and y<edge1[i+1][1]:
                edge1_.append([x,y])
                x+=dx
                y+=dy
        else:
            while x<edge1[i+1][0] and y>edge1[i+1][1]:
                edge1_.append([x,y])
                x+=dx
                y+=dy
                
    edge1_ = np.array(edge1_)
    
    #Defining vertices of image crack
    crack_vertices = np.concatenate((edge0, np.flip(edge1,0))).astype('int')

    #save arrays
    np.save('../data/toy_examples/'+toy_name[:-4]+'_edge0', edge0_)    
    np.save('../data/toy_examples/'+toy_name[:-4]+'_edge1', edge1_)    
    np.save('../data/toy_examples/'+toy_name[:-4]+'_crack_vert', crack_vertices)
    np.save('../data/toy_examples/'+toy_name[:-4]+'_H_params', H_params)            
    
    #Drawing polygon
    cv2.fillPoly(imm, [crack_vertices], (255, 255, 255))

    #Saving toy example
    cv2.imwrite('../data/toy_examples/' + toy_name, imm)
    
    plt.figure()
    plt.imshow(imm)
    plt.plot(edge0_[:,0], edge0_[:,1], 'bo')
    plt.plot(edge1_[:,0], edge1_[:,1], 'ro')
    
#Toy example0
theta = np.pi/64
tx = 10
ty = 5
H_params = np.array([theta,tx,ty])
n_edge0_points = 4
toy_name='toy_example0.png'
create_toy(H_params, n_edge0_points, toy_name=toy_name)

#Toy example1
theta = 0*np.pi/16
tx = 10
ty = 2.5
H_params = np.array([theta,tx,ty])
n_edge0_points = 8
toy_name='toy_example1.png'
create_toy(H_params, n_edge0_points, toy_name=toy_name)

#Toy example2
theta = np.pi/64
tx = +5
ty = -5
H_params = np.array([theta,tx,ty])
n_edge0_points = 3
toy_name='toy_example2.png'
create_toy(H_params, n_edge0_points, toy_name=toy_name) 

#Toy example0
theta = np.pi/64
tx = 10
ty = 5
H_params = np.array([theta,tx,ty])
n_edge0_points = 4
toy_name='toy_example0.png'
create_toy(H_params, n_edge0_points, toy_name=toy_name)

#Toy example1
theta = 0*np.pi/16
tx = 10
ty = 2.5
H_params = np.array([theta,tx,ty])
n_edge0_points = 12
toy_name='toy_example1.png'
create_toy(H_params, n_edge0_points, toy_name=toy_name)

#Toy example2
theta = 0*np.pi
tx = 8
ty = -0
H_params = np.array([theta,tx,ty])
n_edge0_points = 3
toy_name='toy_example2.png'
create_toy(H_params, n_edge0_points, toy_name=toy_name)
