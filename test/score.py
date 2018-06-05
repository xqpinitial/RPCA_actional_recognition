# -*- coding: utf-8 -*-
import numpy as np
import os
import math
import glob
path='result'
label_txt='E_hatlabel.txt'
result1_txt='E_hatresult.txt'
result2_txt='A_hatresult.txt'
num_video=0
acc=0

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]
    return z

for act101 in os.listdir(path):
    for oneact in  os.listdir(path+'/'+act101):
	num_video=num_video+1
	txt_path=path+'/'+act101+'/'+oneact+'/'
	label=np.loadtxt(txt_path+label_txt)
        result0=np.loadtxt(txt_path+result1_txt)
	#result1=np.loadtxt(txt_path+result2_txt)
        #txt_path='result/inception'+'/'+act101+'/'+oneact+'/'
        #result2=np.loadtxt(txt_path+result2_txt)
        result=result0
        #result=2*result0+1.2*(result1+0.8*result2);  #89.1
	#p0=np.multiply((result0-1),(result0-1))
	#p1=np.multiply((result1-1),(result1-1))
	#p2=np.multiply((result1-1),(result1-1))
        #result=result0/p0+result1/p1+result2/p2   #88.2
	if label==np.argmax(result):
	    acc=acc+1
	print txt_path+' result='+str(np.argmax(result))+' label='+str(label)
print 'num of all videos:'+str(num_video)+'\t positive num:'+str(acc)
print '\n acc='+str(100.0*acc/num_video)+'%'
