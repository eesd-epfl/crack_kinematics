#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:09:53 2022

This script contains the codes to measure quantities of the examples published in the
article "Determining crack kinematics from imaged crack patterns" by Pantoja-Rosero et. al

@author: pantoja
"""
import sys
sys.path.insert(1, '../src/')
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2 
import json
import os


def get_tn_tt_from_bin(im_name, data_folder, number_measures):
    """
    Function to measuring values in examples presented in paper
    """
    
    img = cv2.imread('../data/'+ data_folder + im_name+'_pattern.png')
    
    full_kinematics = [np.array([0, number_measures])]
    full_key_points = []
        
    for i in range(number_measures):
        plt.figure()
        plt.imshow(img)
        print('Please click 5 points. 1:Skeleton, 2-3:for tn, 4-5: for tt')
        key_points = np.array(pylab.ginput(5,200))
        print('you clicked:', key_points)
        
        sk = key_points[0]
        tn = np.linalg.norm(key_points[2]-key_points[1])
        
        a = key_points[4]-key_points[3] #vector joining to points initially together
        b = key_points[2]-key_points[1]
        p_ab = np.dot(a,b) / np.linalg.norm(b)
        tt = np.linalg.norm(a-p_ab*b/np.linalg.norm(b))
        
        kin = np.array([tn,tt])
        
        print("the measured kinematic for skeleton point {} is: tn = {}, tt = {}".format(sk,tn,tt))
        
        plt.close()
        
        full_kinematics.append(kin)       
        full_key_points.append(key_points)
        
        
        img = cv2.circle(img, (int(sk[0]),int(sk[1])), radius=3, color=(0, 0, 255), thickness=-1)
    
    full_kinematics = np.array(full_kinematics)
    full_key_points = np.array(full_key_points)
        
    np.save('../results/'+data_folder+im_name+'_kinematic_kps.npy', full_key_points)
    np.save('../results/'+data_folder+im_name+'_kinematic_values.npy', full_kinematics)
    
    cv2.imwrite('../results/'+data_folder+im_name+'_kinematic_skl_pts.jpg', img)

def get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper):
    """
    Function to measuring values in examples presented in paper
    """
    
    if approach=='finite_edges':
        with open('../results/' + data_folder + im_name + '/' + approach + '/eta{}_m{}_l{}_knnor{}_omega{}/crack_kinematic.json'.format(hyper[0],hyper[1],hyper[2],hyper[3],hyper[4]), 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
    else:
        
        with open('../results/' + data_folder + im_name + '/' + approach + '/knnor{}_omega{}/crack_kinematic.json'.format(hyper[0],hyper[1]), 'r') as fp:
            dict_error_kinematic_batch = json.load(fp)
    
    skeleton = []
    for sk in dict_error_kinematic_batch['0']['kinematics_n_t_loc']:
        skeleton.append(sk[0])
    skeleton = np.array(skeleton).reshape((-1,2))
    distances = np.linalg.norm(skeleton-sk_pt, axis = 1)
    min_distance = np.min(distances)
    ind_min_dist = np.where(distances==min_distance)[0]
    sk_pt_read = skeleton[ind_min_dist[0]]
    tn_tt_read = dict_error_kinematic_batch['0']['kinematics_n_t_loc'][ind_min_dist[0]][2]
    skl_pt = dict_error_kinematic_batch['0']['dir_nor_skl'][ind_min_dist[0]]

    print("the kinematic using algorithm for the skeleton point {} is {}".format(sk_pt_read, tn_tt_read))

    return tn_tt_read, skl_pt#, t_dofs_loc    


def get_tn_tt_from_bin_2pts(im_name, data_folder, approach, hyper, number_measures):
    """
    Function to measuring values in examples presented in paper
    """
    
    img = cv2.imread('../data/'+ data_folder + im_name+'_pattern.png')
    
    full_kinematics = [np.array([0, number_measures])]
    full_key_points = []
        
    for i in range(number_measures):
        plt.figure()
        plt.imshow(img)
        print('Please click 3 points. 1:Skeleton, 2-3:for tn-tt')
        key_points = np.array(pylab.ginput(3,200))
        print('you clicked:', key_points)
        plt.close()
        sk = key_points[0]
        
        _, skl_pt = get_tn_tt_from_results(im_name, data_folder, sk, approach, hyper)
        #Computing kinematics normal tangential for local values
        
        #case II -- I and IV quartier
        beta = (skl_pt[6] if skl_pt[6]>0 else skl_pt[6]+180)*np.pi/180 #if the angle is negative, then add 180. This as n axis is pointing always in the I or II quartier.
        transf_matrix_skl_pt_loc = np.array([[np.cos(beta), np.sin(beta)],[-np.sin(beta), np.cos(beta)]])
        
        
        t_local = (key_points[2]-key_points[1]).reshape((2,-1))
        kin_n_t_loc = (transf_matrix_skl_pt_loc @ t_local).reshape(-1)
        kin = np.array(kin_n_t_loc)
        
        print("the measured kinematic for skeleton point {} is: tn = {}, tt = {}".format(sk,abs(kin[0]),abs(kin[1])))
        
        full_kinematics.append(kin)       
        full_key_points.append(key_points)
        
        
        img = cv2.circle(img, (int(sk[0]),int(sk[1])), radius=3, color=(0, 0, 255), thickness=-1)
    
    full_kinematics = np.array(full_kinematics)
    full_key_points = np.array(full_key_points)
        
    np.save('../results/'+data_folder+im_name+'_kinematic_kps.npy', full_key_points)
    np.save('../results/'+data_folder+im_name+'_kinematic_values.npy', full_kinematics)
    
    cv2.imwrite('../results/'+data_folder+im_name+'_kinematic_skl_pts.jpg', img)
    

def plot_kin_line(im_name, full_key_points, full_kinematics, number_measures):
    
    img = cv2.imread('../data/'+ data_folder + im_name+'_pattern.png')
    
    for i in range(number_measures):
        
        pts = full_key_points[i][1:].astype('int32')
        img = cv2.line(img, (pts[0][0],pts[0][1]), (pts[1][0], pts[1][1]), color=(0, 0, 255), thickness=2)
    
    cv2.imwrite('../results/'+data_folder+im_name+'_kinematic_skl_line.jpg', img)


#User interactionreal_patterns
#Comparison gauge mesurement for beam sec 4.6

#PATTERN1
#beam_patch0_y627_x2613_mask__IMG_20220509_105555_mask
data_path = "../data/beam/patterns/pattern1/"        
mask_name = "beam_patch0_y627_x2613_mask__IMG_20220509_105555_mask.png" 
im_name = 'beam_patch0_y627_x2613_mask__IMG_20220509_105555_mask'
im_path = 'beam_patch0_y627_x2613_IMG_20220509_105555.jpg'
data_folder = 'pattern1/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0419)


#beam_patch1_y1551_x2127_mask__IMG_20220509_105555_mask
data_path = "../data/beam/patterns/pattern1/"        
mask_name = "beam_patch1_y1551_x2127_mask__IMG_20220509_105555_mask.png" 
im_name = 'beam_patch1_y1551_x2127_mask__IMG_20220509_105555_mask'
im_path = 'beam_patch1_y1551_x2127_IMG_20220509_105555.jpg'
data_folder = 'pattern1/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0419)



#beam_patch2_y2008_x1956_mask__IMG_20220509_105555_mask
data_path = "../data/beam/patterns/pattern1/"        
mask_name = "beam_patch2_y2008_x1956_mask__IMG_20220509_105555_mask.png" 
im_name = 'beam_patch2_y2008_x1956_mask__IMG_20220509_105555_mask'
im_path = 'beam_patch2_y2008_x1956_IMG_20220509_105555.jpg'
data_folder = 'pattern1/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0419)


#beam_patch3_y2717_x2004_mask__IMG_20220509_105555_mask
data_path = "../data/beam/patterns/pattern1/"        
mask_name = "beam_patch3_y2717_x2004_mask__IMG_20220509_105555_mask.png" 
im_name = 'beam_patch3_y2717_x2004_mask__IMG_20220509_105555_mask'
im_path = 'beam_patch3_y2717_x2004_IMG_20220509_105555.jpg'
data_folder = 'pattern1/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0419)


#beam_patch4_y3186_x1877_mask__IMG_20220509_105555_mask
data_path = "../data/beam/patterns/pattern1/"        
mask_name = "beam_patch4_y3186_x1877_mask__IMG_20220509_105555_mask.png" 
im_name = 'beam_patch4_y3186_x1877_mask__IMG_20220509_105555_mask'
im_path = 'beam_patch4_y3186_x1877_IMG_20220509_105555.jpg'
data_folder = 'pattern1/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0419)


###PATTERN2
#beam_patch0_y655_x2589_mask__IMG_20220509_105611_mask
data_path = "../data/beam/patterns/pattern2/"        
mask_name = "beam_patch0_y655_x2589_mask__IMG_20220509_105611_mask.png" 
im_name = 'beam_patch0_y655_x2589_mask__IMG_20220509_105611_mask'
im_path = 'beam_patch0_y655_x2589_IMG_20220509_105611.jpg'
data_folder = 'pattern2/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0425)

#beam_patch1_y1268_x2601_mask__IMG_20220509_105611_mask
data_path = "../data/beam/patterns/pattern2/"        
mask_name = "beam_patch1_y1268_x2601_mask__IMG_20220509_105611_mask.png" 
im_name = 'beam_patch1_y1268_x2601_mask__IMG_20220509_105611_mask'
im_path = 'beam_patch1_y1268_x2601_IMG_20220509_105611.jpg'
data_folder = 'pattern2/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0425)

#beam_patch2_y1706_x2692_mask__IMG_20220509_105611_mask
data_path = "../data/beam/patterns/pattern2/"        
mask_name = "beam_patch2_y1706_x2692_mask__IMG_20220509_105611_mask.png" 
im_name = 'beam_patch2_y1706_x2692_mask__IMG_20220509_105611_mask'
im_path = 'beam_patch2_y1706_x2692_IMG_20220509_105611.jpg'
data_folder = 'pattern2/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0425)

#beam_patch3_y2386_x2191_mask__IMG_20220509_105611_mask
data_path = "../data/beam/patterns/pattern2/"        
mask_name = "beam_patch3_y2386_x2191_mask__IMG_20220509_105611_mask.png" 
im_name = 'beam_patch3_y2386_x2191_mask__IMG_20220509_105611_mask'
im_path = 'beam_patch3_y2386_x2191_IMG_20220509_105611.jpg'
data_folder = 'pattern2/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0425)

#beam_patch4_y2661_x2740_mask__IMG_20220509_105611_mask
data_path = "../data/beam/patterns/pattern2/"        
mask_name = "beam_patch4_y2661_x2740_mask__IMG_20220509_105611_mask.png" 
im_name = 'beam_patch4_y2661_x2740_mask__IMG_20220509_105611_mask'
im_path = 'beam_patch4_y2661_x2740_IMG_20220509_105611.jpg'
data_folder = 'pattern2/'
number_measures = 1
hyper = [1.0,1.0,4,10,0.0]
approach = "finite_edges"
#Select sk_pt
img = cv2.imread(data_path+mask_name)
img_rgb = cv2.imread(data_path+im_path, cv2.COLOR_BGR2RGB)
if os.path.isfile('../results/'+data_folder+'sk_pt_'+mask_name+'.npy'):
    sk_pt = np.load('../results/'+data_folder+'sk_pt_'+mask_name+'.npy')
else:
    plt.figure()
    #plt.imshow(img, 'gray')
    plt.imshow(img_rgb)
    print('Please click 1 points for skeleton')
    sk_pt = np.array(pylab.ginput(1,200))
    print('you clicked:', sk_pt)    
    plt.close()
    sk_pt = sk_pt[0]
    np.save('../results/'+data_folder+'sk_pt_'+mask_name+'.npy', sk_pt)
print(im_name + " ---------------")
img = cv2.circle(img, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
img_rgb = cv2.circle(img_rgb, (int(sk_pt[0]),int(sk_pt[1])), radius=3, color=(0, 0, 255), thickness=-1)
cv2.imwrite('../results/'+data_folder+'sk_pt_'+mask_name, img)
cv2.imwrite('../results/'+data_folder+'sk_pt_rgb'+mask_name, img_rgb)
tn_tt_read,_ = get_tn_tt_from_results(im_name, data_folder, sk_pt, approach, hyper)
print("result in mm is ", np.array(tn_tt_read)*.0425)