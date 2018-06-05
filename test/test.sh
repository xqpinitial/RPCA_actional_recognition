#! /user/bin sh

/caffe test   \
 --weights=models/ucf101_split1_rpca_resnet50_iter_9000.caffemodel  \
--model=bn_inception_rpca_train_val.prototxt \
-gpu 0 -iterations 100 2>&1 | tee test.log


