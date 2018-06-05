'''
A sample function for classification using temporal network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import glob
import os
import numpy as np
import math
import cv2
import scipy.io as sio
import random

test_num=25
frame_scores = np.zeros((101,30))
def Videorpca_E_HAT_rediction(
        vid_name,
        net,
        num_categories,
        feature_layer,
        start_frame=0,
        num_frames=0,
        num_samples=3,
        stack_frames=1
        ):
    imglist = glob.glob(os.path.join(vid_name, '*.jpg'))
    duration = len(imglist)
    # selection
    step = int(math.floor((duration-stack_frames+1)/num_samples))
    dims = (256,256,3,num_samples)
    rpca_ = np.zeros(shape=dims, dtype=np.float64)
    rpca_flip_ = np.zeros(shape=dims, dtype=np.float64)
    

    for num in range(test_num):
        start_frame=start_frame+1;
        start_frame=start_frame%step
        for i in range(num_samples):
            for j in range(stack_frames):
                rpca_file = os.path.join(vid_name, '{0:0d}.jpg'.format(i*step+j+1 + start_frame))
                #print rpca_file
                rpca_img = cv2.imread(rpca_file, 1)
	        if rpca_img is None:
		    print rpca_file+"is None"
	            continue
                rpca_img = cv2.resize(rpca_img, dims[1::-1])
	        rpca_[:,:,3*j:3*(j+1),i] = rpca_img
                rpca_flip_[:,:,3*j:3*(j+1),i] = rpca_img[:,::-1,:]
    	# crop
    	rpca_1 = rpca_[:224, :224, :,:]
	rpca_2 = rpca_[:224, -224:, :,:]
    	rpca_3 = rpca_[16:240, 16:240, :,:]
    	rpca_4 = rpca_[-224:, :224, :,:]
	rpca_5 = rpca_[-224:, -224:, :,:]
    	rpca_f_1 = rpca_flip_[:224, :224, :,:]
    	rpca_f_2 = rpca_flip_[:224, -224:, :,:]
    	rpca_f_3 = rpca_flip_[16:240, 16:240, :,:]
    	rpca_f_4 = rpca_flip_[-224:, :224, :,:]
    	rpca_f_5 = rpca_flip_[-224:, -224:, :,:]
    	rpca = np.concatenate((rpca_1,rpca_2,rpca_3,rpca_4,rpca_5,rpca_f_1,rpca_f_2,rpca_f_3,rpca_f_4,rpca_f_5), axis=3)
    	# substract mean
    	d = sio.loadmat('mean128.mat')
    	rpca_mean = d['mean128']
    	rpca = rpca - np.tile(rpca_mean[...,np.newaxis], (1, 1, 1, rpca.shape[3]))
    	rpca = np.transpose(rpca, (1,0,2,3))

    	# test
    	batch_size = 30
    	prediction = np.zeros((num_categories,rpca.shape[3]))
    	num_batches = int(math.ceil(float(rpca.shape[3])/batch_size))

    	for bb in range(num_batches):
            span = range(batch_size*bb, min(rpca.shape[3],batch_size*(bb+1)))
            net.blobs['data'].data[...] = np.transpose(rpca[:,:,:,span], (3,2,1,0))
            output = net.forward()
            prediction[:, span] = np.transpose(output[feature_layer])
	    global frame_scores
	    if num==1:
		frame_scores=prediction
        frame_scores=np.dstack((frame_scores,prediction))

    return frame_scores











