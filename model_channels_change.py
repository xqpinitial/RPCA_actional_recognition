__author__ = 'Yuanjun Xiong'
"""
This script will transform an image based Caffe model to its optic flow ready form
The basic approach is to average the three channels of the first set of convolution filters.
Averaged filters are then replicated K times to incorporate K input frames of optical flow maps.
Refer to "Towards Good Practices for Very Deep Two-Stream ConvNets" for more details.
======================================================================
Usage:
    python build_flow_network.py <caffe root> <first layer name> <image model prototxt> <image model weights> <flow model prototxt> <flow model weights[out]>
You need to edit the flow model prototxt manually (to have 10 channels input) before running this script.
======================================================================
This script is released for unlimited use.
Yuanjun Xiong
@MMLAB, CUHK
Nov 13, 2015
"""


import sys

CAFFE_ROOT = sys.argv[1]
LAYER_NAME = sys.argv[2]
SRC_NET = sys.argv[3]
SRC_WEIGHTS = sys.argv[4]
TGT_NET = sys.argv[5]
TGT_WEIGHTS = sys.argv[6]


sys.path.append(CAFFE_ROOT+'/python')
import caffe

net = caffe.Net(SRC_NET, SRC_WEIGHTS, caffe.TEST)
target_net = caffe.Net(TGT_NET, caffe.TEST)

trans_layer_name = LAYER_NAME
conv1_data = net.params[trans_layer_name][0].data

# take mean filters
m_c1_d = conv1_data.mean(axis=1) 

#replicate conv1 params
for i in xrange(target_net.params[trans_layer_name][0].data.shape[1]):
    target_net.params[trans_layer_name][0].data[:, i, :, :] = m_c1_d

target_net.params[trans_layer_name][1].data[:] = net.params[trans_layer_name][1].data

#copy other weights
for name in net.params.keys()[1:]:
    if name in target_net.params:
        for i in xrange(len(target_net.params[name])):
            target_net.params[name][i].data[:] = net.params[name][i].data

#dump the target weights
target_net.save(TGT_WEIGHTS)