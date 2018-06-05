#!/usr/bin/env python

'''
A sample script to run classificition using both spatial/temporal nets.
Modify this script as needed.
'''
import sys
sys.path.append('caffe/python')
import numpy as np
import caffe
import math
import glob
import os
import math
import cv2
import scipy.io as sio

from Videorpca_E_HAT_rediction import Videorpca_E_HAT_rediction

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():
    caffe.set_mode_gpu()
    out_path='result'
    # temporal prediction
    model_def_file = '../res50/UCF101/resnet50_deploy.prototxt'
    model_file = '../res50/UCF101/UCF101_rpca_resnet50_iter_4000.caffemodel'
    print "load model"
    temporal_net = caffe.Net(model_def_file, model_file, caffe.TEST)
    print "load model done"
    # input video (containing image_*.jpg and flow_*.jpg) and some settings
    test_videos = '../../make_label/UCF101/e_hat_test.txt'
    start_frame = 0
    num_categories = 101
    feature_layer = 'fc-action'
    num_video=0
    # temporal net prediction
    
    test_file=np.loadtxt(test_videos,str,'\n')
    for i_dir_num in test_file:
	num_video=num_video+1
	print "testing video num="+str(num_video)
	test_videos_dir=i_dir_num[0]
	label=i_dir_num[2]
    	result_path =out_path+ test_videos_dir[30:]
        if (not os.path.exists(result_path)):	
    	    os.makedirs(result_path)
    	prediction = Videorpca_E_HAT_rediction(
            test_videos_dir,
            temporal_net,
            num_categories,
            feature_layer,
            start_frame)
	avg_pred = np.mean(prediction, axis=1)
        avg_pred = np.mean(avg_pred, axis=1)
        print avg_pred.shape
    	avg_pred = softmax(avg_pred)
	np.savetxt(result_path+'result.txt',avg_pred)
	label_txt=open(result_path+"label.txt",'w')
        label_txt.write(label)
        label_txt.close()



if __name__ == "__main__":
    main()
