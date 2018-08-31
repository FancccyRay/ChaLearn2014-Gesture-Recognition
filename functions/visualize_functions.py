
from gzip import GzipFile
from cPickle import dump, load
import os

import matplotlib.pyplot as pl
#import pylab as pl
from math import sqrt, ceil, floor
import numpy as n

def gray_body_conv1():
    print " draw gray_body_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[0]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]

    filters = W[:, 0, 0, :, :,]


    _save_path = r'/home/fancy/Desktop/sys_project/temp/visualization/gray_body_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def gray_hand_conv1():
    print " draw gray_body_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[0]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]
    
    filters = W[:, 1, 0, :, :,]
    _save_path = r'/home/fancy/Desktop/sys_project/temp/visualization/gray_hand_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def depth_body_conv1():
    print " draw depth_body_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[2]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]

    filters = W[:, 0, 0, :, :,]

    _save_path = r'/home/fancy/Desktop/sys_project/temp/visualization/depth_body_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def depth_hand_conv1():
    print " draw depth_hand_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[2]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]
    
    filters = W[:, 1, 0, :, :,]
    _save_path = r'/home/fancy/Desktop/sys_project/temp/visualization/depth_hand_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def draw_kernel(filters, _save_path):
    
    filter_start = 0
    fignum = 0
    num_filters = filters.shape[0]
    filters = (filters - filters.min())/(filters.max()-filters.min())

    FILTERS_PER_ROW = 16
    MAX_ROWS = 16
    MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS

    f_per_row = FILTERS_PER_ROW 
    filter_end = min(filter_start+MAX_FILTERS, num_filters)
    filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
    filter_size = int((filters.shape[1]))
    fig = pl.figure(fignum)
    #fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
    num_filters = filter_end - filter_start

    bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)

    for m in xrange(filter_start,filter_end ):
        filter_pic = filters[m, :,:]
        y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
        bigpic[ 1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
    pl.xticks([])
    pl.yticks([])

    pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
    pl.savefig(_save_path, format='eps',bbox_inches='tight')
    pl.show()


def draw_original():
    print " draw original"
    _save_path = r'visualization\\original.eps'

    pc = "wudi"
    if pc=="wudi":
        src = r"I:\Kaggle_multimodal\Training_prepro"
        res_dir_ = r"/home/fancy/Desktop/sys_project/temp/visualization/"# dir of original data -- note that wudi has decompressed it!!!
    elif pc=="lio":
        src = "/mnt/wd/chalearn/preproc"
        res_dir_ = "/home/lpigou/chalearn_wudi/try"

    import h5py
    from math import floor
    file = h5py.File(src+"/data%d.hdf5", "r", driver="family", memb_size=2**32-1)
    x_train = file["x_train"]
    x_valid = file["x_valid"]
    y_train = file["y_train"]
    y_valid = file["y_valid"]

    # which frame to plot can be changed inside the function
    # here we have chosen a random frame to plot
    # because we save it as h5py, top 1000 frames is chosen as random

    frame_to_plot = ceil(n.random.rand() *1000 )
    images = x_train[frame_to_plot,:,:,:,:,:]
    
    f_per_row = images.shape[2]
    filter_rows = images.shape[0] * images.shape[1]   
    num_filters = images.shape[0] * images.shape[1] * images.shape[2]
    filter_size = images.shape[-1]    

    bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)

    for rgb_depth in range(images.shape[0]):
        for body_hand in range(images.shape[1]):
            for frame_num in range(images.shape[2]):
                filter_pic = images[rgb_depth, body_hand, frame_num, :,:]
                x = frame_num
                y = rgb_depth * 2  + body_hand
                bigpic[ 1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                        1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
    pl.xticks([])
    pl.yticks([])
    pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
    pl.savefig(_save_path, format='eps', bbox_inches='tight')
    pl.show()

def plot_confusion_matrix():
    """"plot the confusion matrix"""
    from sklearn.metrics import confusion_matrix
    from numpy import ones, array, prod, zeros, empty, inf, float32, random
    import numpy
    import os
    import zipfile
    import shutil
    import csv
    import re
    print "plot confusion matrix"
    _save_path = r'/home/fancy/Desktop/sys_project/temp/visualization/cm.eps'
    truth_dir=r'/home/fancy/Desktop/sys_project/test_set'
    prediction_dir=r'/home/fancy/Desktop/sys_project/Test_early_fusion_pred'
    # Get the list of samples from ground truth
    gold_list = os.listdir(truth_dir)

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;

    begin_add=0
    end_add=0

    for gold in gold_list:
        # Avoid double check, use only labels file
        if not gold.lower().endswith("_labels.csv"):
            continue

        # Build paths for prediction and ground truth files
        sampleID=re.sub('\_labels.csv$', '', gold)
        labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
        dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
        predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")
                # Get the number of frames for this sample
        with open(dataFile, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                seqlenght=int(row[0])
            del filereader
        # Get the number of frames for this sample
        """ Evaluate this sample agains the ground truth file """
        maxGestures=20

        # Get the list of gestures from the ground truth and frame activation
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqlenght))
        with open(labelsFile, 'rb') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))

        # Get the list of gestures from prediction and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqlenght))
        with open(predFile, 'rb') as csvfilepred:
            csvpred = csv.reader(csvfilepred)
            for row in csvpred:
                binvec_pred[int(row[0])-1, int(row[1])-1+begin_add:int(row[2])-1+end_add] = 1
                predGestures.append(int(row[0]))

        # Get the list of gestures without repetitions for ground truth and predicton
        gtGestures = numpy.unique(gtGestures)
        predGestures = numpy.unique(predGestures)

        # Find false positives
        falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

        # Get overlaps for each gesture
        overlaps = numpy.zeros(maxGestures)
        for idx in gtGestures:
            intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
            aux = binvec_gt[idx-1] + binvec_pred[idx-1]
            union = sum(aux > 0)
            overlaps[idx-1] = intersec/union
        
        print len(overlaps)
        # Use real gestures and false positive gestures to calculate the final score
   
    
    cm = confusion_matrix(y_test, y_pred)
    # Show confusion matrix in a separate window
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.savefig(_save_path, format='eps', bbox_inches='tight')
    pl.show()


def plot_cnn_error_rate():
    import numpy
    _save_path = r'/home/fancy/Desktop/sys_project/temp/visualization/training_error.eps'

    validation_error = numpy.array([ 70.918,  69.277, 67.949, 67.344, 
    68.789,  67.285,  67.793,  67.734,  67.422,  66.855])
    validation_error /=100
    training_error = numpy.array([71.537, 57.717, 51.198,  45.355,
    40.530, 36.164, 31.002, 27.170, 24.192, 21.479])
    training_error /= 100

    training_cost = numpy.array([3.000, 1.799, 1.492, 1.298, 1.132, 0.991,
    0.863, 0.751, 0.658, 0.592])
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    plt.clf()
    plt.plot(range(validation_error.shape[-1]), validation_error, color='c',linewidth=2.0, label="validation error")
    plt.plot(range(training_error.shape[-1]), training_error, color='r',linewidth=2.0, label="training error" )
    plt.plot(range(training_cost.shape[-1]), training_cost, color='g',linewidth=2.0, label="training cost")
    plt.legend(prop={'size':20})
    plt.ylabel('frame error rate',  fontsize=20)
    plt.xlabel('epoches',  fontsize=20)
    from pylab import savefig
    savefig(_save_path, format='eps', bbox_inches='tight')
    plt.show()

def plot_sk_error_rate():
    import numpy
    _save_path = r'/home/fancy/Desktop/sys_project/temp/visualization/training_error_sk.eps'
    validation_error = numpy.load('/home/fancy/Desktop/sys_project/temp/result/validation_loss.npy')
    training_error = numpy.load('/home/fancy/Desktop/sys_project/temp/result/minibatch_avg_cost_train.npy')

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)
    plt.clf()
    plt.plot(range(validation_error.shape[-1]), validation_error, color='c',linewidth=2.0, label="validation error")
    #plt.plot(range(training_error.shape[-1]), training_error, color='r',linewidth=2.0, label="training cost: negative loglikelihood")
    plt.legend(prop={'size':20})
    plt.ylabel('frame error rate', fontsize=20)
    plt.xlabel('epoches', fontsize=20)
    from pylab import savefig
    savefig(_save_path, format='eps', bbox_inches='tight')
    plt.show()