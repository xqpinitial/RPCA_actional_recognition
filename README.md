# RPCA_actional_recognition
In human action recognition, visual feature descriptors are of vital importance. We propose a novel thought of getting the low-rank features and sparse features from the video. This ideal is based on the Robust Principal Component Analysis (RPCA), and obtained through sparse component and low-rank on the raw image pixels of each video. Then we use the two-stream architecture for our robust PCA segmentation descriptors with fine-tuning.

1、using caffe https://github.com/yjxiong/caffe/tree/action_recog

2、git clone https://github.com/xqpinitial/Robust-PCA-RPCA <br>
   the v_BoxingPunchingBag_g01_c02.rar is demo of rpca <br>
   unrar PROPACK.rar  and run video_rpca_main.m to get rpca features <br>
   ![](https://github.com/xqpinitial/RPCA_actional_recognition/raw/master/Screenshots/rpca.jpg)  

3、train A_hat_C3(low-rank feature) and E_hat_C3(sparse feature) <br>
 using model_channels_change.py to revise your model for train
