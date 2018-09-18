# ChaLearn2014-Gesture-Recognition
Take 3DCNN and DBN as sub-networks, merge two networks using late fusion algorithm

"""
The project is copied from https://github.com/stevenwudi/chalearn2014_wudi_lio 
I met some questions when running the project, so I made some adjustment here
To be continue...
"""

You can conduct the file as the following order,then you can get the training result of DBN/3DCNN/
Multi-model network.

step 1:Step_1_preprocess.py 
----
extract skeleton features & preprocess videos 

step 2:Step_2_preproc_save_as_hdf5_file.py
----------
save as hdf5 file to read more easily

step 3: Step_3_DBN_training.py
--------
(vedio classifier) construct a GDBN networkas and train it to test its classification quality
(train this network twice)        
（1）¢pretrain the network and update parameters   
(2)¢load the pretrained parameters and finetune the network

step 4: Step_4_3DCNN_training.py
----
(vedio classifier) construct a 3DCNN networkas and train it to test its classification quality,     
if the training stop due to some problems,you can load the parameteres that we trained just now,
modify the epoches,use according to your need

step5: Step_5_DBN_3DCNN_traing_fusion_net.py
--
video classifier using a 3DCNN and DBN, fusing the two features form submodel together, 
load dbn&3dcnn parameters trained by sub model seperately 

These Three Steps Can Be Done At The Same Time 
----
step6_1: Step_6_1_DBN__test_state_matrix.py
train the DBN network using test set, we get the state matrix of the data 

step6_2: Step_6_2_3DCNN_test_state_matrix.py
train the 3DCNN using test set, we get the state matrix of the data 

step6_3: Step_6_3_fusion_network_test_state_matrix.py
train the mult-model network using test set, we get the state matrix of the data 


step7: Step_7_test_result.py
-----
compute the state matrix to get predicted labels,test the classification quality using 
viterbi decoding and 
compute the score of predicted data using Jaccard Index

step8: Step_final_visualization.py
-------
draw the error rate,confusion matrix 
