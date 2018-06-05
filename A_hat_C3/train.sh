#! /user/bin sh
../caffe train --solver=bn_inception_rpca_solver.prototxt  --weights=initmodel/resnet50.caffemodel 2>&1 | tee ucf101_rpca_ahat_resnet.log 


