#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:42:00 2018

@author: fancy
"""

import cv2
import os
import sys
import shutil
import errno
import gzip
from time import time
from itertools import tee, islice
from cPickle import dump
from glob import glob
from random import shuffle
from numpy import *
from numpy.random import RandomState
from classes import GestureSample
from functions.preproc_functions import *

#data path and store path definition
#pc = "win_fancy"
pc = "linux_fancy"
if pc == "linux_fancy":
    data = "/home/fancywu/Desktop/GestureRecognition/Train_files/Test_Samples"  # dir of original data
elif pc == "win_fancy":
    data = ""


# global variable definition
show_gray = False
show_depth = False
show_user = False
# 640 x 480 video resolution
vid_res = (480, 640)  
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft', 'HandLeft',
               'ElbowRight', 'WristRight', 'ShoulderRight', 'HandRight',
               'Head', 'Spine', 'HipCenter']

# Then we  choose 8 frame before and after the ground true data:
NEUTRUAL_SEG_LENGTH = 8
# number of hidden states for each gesture class
HIDDEN_STATE = 5

prog_start_time = 0
def main():
    prog_start_time = time()
    os.chdir(data)
    samples = glob("*")  #scan and get all samples
    print len(samples), "samples found"
    #start preprocessing
    preprocess(samples)
#    print"whole programe time used: %3.3f h"%((time()- prog_start_time)/3600.)

def preprocess(samples):
    for file_count, file in enumerate(sort(samples)):
        print "This is the %d th file : " %file_count
        ##totally 400 training samples, use 36 for training, 40 for validating
        if file_count < 361:
            if pc == "linux_fancy" :
                dest = "/home/fancywu/Desktop/GestureRecognition/Train_files/Preprocessed_files/train"
            elif pc == "win_fancy" :
                dest =""
            print "\t Processing training file " + file
            
        elif file_count >= 361:
            if pc == "linux_fancy" :
                dest = "/home/fancywu/Desktop/GestureRecognition/Train_files/Preprocessed_files/valid"
            elif pc == "win_fancy" :
                dest =""
            print "\t Processing validating file " + file
        
        start_time = time()
	    ##Create the object to access the sample
        sample = GestureSample(os.path.join(data, file))
#        print(os.path.join(data, file))
    
        ##USE Ground Truth information to learn the model
        ##Get the list of gesture for this sample
        gestures = sample.getGestures()
        print "len(gestures" + str(len(gestures))
        # preprocess each gesture 
        for gesture in gestures:
            skelet, depth, gray, user, c = sample.get_data_wudi(gesture, vid_res, NEUTRUAL_SEG_LENGTH)
            if c: print '1: corrupt'
            
            skelet_feature, Targets, c = proc_skelet_wudi(sample, used_joints, gesture, HIDDEN_STATE,
                                                              NEUTRUAL_SEG_LENGTH)
            if c: print '2: corrupt'
            
            user_o = user.copy()
            user = proc_user(user)
            skelet, c = proc_skelet(skelet)
            if c: print '3: corrupt'
            
            user_new, depth, c = proc_depth_wudi(depth, user, user_o, skelet, NEUTRUAL_SEG_LENGTH)
            if c: print '4: corrupt'
            
            gray, c = proc_gray_wudi(gray, user, skelet, NEUTRUAL_SEG_LENGTH)
            if c: print '5: corrupt'

            if show_depth: play_vid_wudi(depth, Targets, wait=1000 / 10, norm=False)
            if show_gray: play_vid_wudi(gray, Targets, wait=1000 / 10, norm=False)
            if show_user: play_vid_wudi(user_new, Targets, wait=1000 / 10, norm=False)
            
            traj2D, traj3D, ori, pheight, hand, center = skelet
#            print len(skelet),skelet
            skelet = traj3D, ori, pheight
#            print len(skelet),skelet
            
            assert user.dtype == gray.dtype == depth.dtype == traj3D.dtype == ori.dtype == "uint8"
            assert gray.shape == depth.shape
#            print "gray.shape=" , gray.shape
            
            if not gray.shape[1] == skelet_feature.shape[0] == Targets.shape[0]:
                print "too early or too late movement, skip one"
            print "gray.shape[1]=" , gray.shape[1]
            
            ##we just use gray and depth videos for training
            video = empty((2,) + gray.shape, dtype = "uint8")   
            video[0], video[1] = gray, depth
            store_preproc_video_skelet_data(video, skelet_feature, Targets.argmax(axis = 1), skelet, dest)
            print "finished"
            
        print "Processing one batch requires : %d seconds\n" %(time() - start_time)
        if file_count == len(samples) - 1:
            dump_last_data(video, skelet_feature, Targets.argmax(axis = 1), skelet, dest)
        if file_count == 361 - 1:
            dump_last_data(video, skelet_feature, Targets.argmax(axis = 1), skelet, dest)
            
        print" Processing one sample requies: %3.3f h" %((time()- prog_start_time)/3600.)
        
        
    
    
    

vid, skel_fea, labl, skel = [], [], [], [] 
count = 1
batch_idx = 0
def store_preproc_video_skelet_data(video, skelet_feature, label, skelet, dest_path):
    global vid, skel_fea, labl, skel, count, batch_idx
    if len(vid) == 0:
        vid = video
        skel_fea = skelet_feature 
        labl = label
        skel.append(skelet)
    else:
        vid = concatenate((vid,video), axis = 2)
        skel_fea = concatenate((skel_fea,skelet_feature), axis = 0)
        labl = concatenate((labl,label))
        skel.append(skelet)

    if len(labl) > 1000:
        make_sure_path_exists(dest_path)
        os.chdir(dest_path)
        file_name = "batch__" + str(batch_idx) + "_" + str(len(labl)) + ".zip"
        file = gzip.GzipFile(file_name, 'wb')
        dump((vid, skel_fea, labl, skel), file, -1)
        file.close()
        
        print file_name
        batch_idx += 1
        count = 1
        vid, skel_fea, labl, skel = [], [], [], [] 
    
    count += 1
        
            
def dump_last_data(video, skelet_feature, label, skelet, dest_path):
    global vid, skel_fea, labl, skel, count, batch_idx
    vid = concatenate((vid,video), axis = 2)
    skel_fea = concatenate((skel_fea,skelet_feature), axis = 0)
    labl = concatenate((labl, label))
    skel.append(skelet)
    os.chdir(dest_path)
    file_name = "batch+" + str(batch_idx) + "_" + str(len(labl)) + ".zip"
    file = gzip.GzipFile(file_name, 'wb')
    dump((vid, skel_fea, labl, skel), file, -1)
    file.close()
    
    print file_name
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
            
                
                
                
            
            
