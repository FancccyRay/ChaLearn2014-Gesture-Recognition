# various imports
from cPickle import load
from glob import glob
from time import time, localtime
from gzip import GzipFile
import os
# numpy imports
from numpy import zeros, empty, inf, float32, random

# theano imports
from theano import function, config, shared
import theano.tensor as T

# customized imports
from dbn.GRBM_DBN import GRBM_DBN
from conv3d_chalearn import conv3d_chalearn
from convnet3d import LogRegr

#  modular imports
# the hyperparameter set the data dir, use etc classes, it's important to modify it according to your need
from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop,\
                                     net ,  DataLoader_with_skeleton_normalisation
from functions.train_functions import _shared, _avg, write, ndtensor, print_params, lin,\
                                      training_report, epoch_report, _batch,\
                                      save_results, move_results, save_params, test_lio_skel

prog_start_time = time()
####################################################################
####################################################################
print "\n%s\n\t initializing \n%s"%(('-'*30,)*2)
####################################################################
####################################################################
# source and result directory
#pc = "fancy_win"
pc = "fancy_linux"
if pc=="fancy_linux":
    src = r"/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/hdf5Dest/"
    res_dir_ = r"/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/result/3dcnn/"# dir of original data 
elif pc == "fancy_win":
    src = r""
    res_dir_ = r""

lt = localtime()
res_dir = res_dir_+"/try/"+str(lt.tm_year)+"."+str(lt.tm_mon).zfill(2)+"." \
            +str(lt.tm_mday).zfill(2)+"."+str(lt.tm_hour).zfill(2)+"."\
            +str(lt.tm_min).zfill(2)+"."+str(lt.tm_sec).zfill(2)
os.makedirs(res_dir)
#  global variables/constants
# ------------------------------------------------------------------------------
params = [] # all neural network parameters
layers = [] # all architecture layers
mini_updates = []
micro_updates = []
last_upd = []
update = []


# shared variables
learning_rate = shared(float32(lr.init))
if use.mom: 
    momentum = shared(float32(mom.momentum))
    drop.p_vid = shared(float32(drop.p_vid_val) )
    drop.p_hidden = shared(float32(drop.p_hidden_val))


idx_mini = T.lscalar(name="idx_mini") # minibatch index
idx_micro = T.lscalar(name="idx_micro") # microbatch index
x = ndtensor(len(tr.in_shape))(name = 'x') # video input
y = T.ivector(name = 'y') # labels
x_ = _shared(empty(tr.in_shape))
y_ = _shared(empty(tr.batch_size))
y_int32 = T.cast(y_,'int32')


L1 = _shared(0)
L2 = _shared(0)


### useless fake, but DataLoader_with_skeleton_normalisation would require that
x_skeleton = ndtensor(len(tr._skeleon_in_shape))(name = 'x_skeleton') # video input
x_skeleton_ = _shared(empty(tr._skeleon_in_shape))

# load the skeleton normalisation --Lio didn't normalise video input, but should we?
import cPickle
f = open('CNN_normalization.pkl','rb')
CNN_normalization = cPickle.load(f)
Mean_CNN = CNN_normalization ['Mean_CNN']
Std_CNN = CNN_normalization['Std_CNN']

# customized data loader for both video module and skeleton module
loader = DataLoader_with_skeleton_normalisation(src, tr.batch_size, Mean_CNN, Std_CNN) # Lio changed it to read from HDF5 files

####################################################################
# 3DCNN for video module
####################################################################
#if u have already trained the network before, the might pause for some reason, u can load the updated paramters 
# we load the CNN parameteres here
#use.load = True
#load_path = '/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/result_temp/3dcnn/try/70.9% 2018.05.05.02.28.21/' 
#video_cnn = conv3d_chalearn(x, use, lr, batch, net, reg, drop, mom, tr, res_dir,load_path)
video_cnn = conv3d_chalearn(x, use, lr, batch, net, reg, drop, mom, tr, res_dir)

#####################################################################
# fuse the ConvNet output with skeleton output  -- need to change here
######################################################################  
out = video_cnn.out
# some activation inspection
insp =  []
for insp_temp in video_cnn.insp_mean:    insp.append(insp_temp)
insp = T.stack(insp)

# softmax layer
layers.append(LogRegr(out, rng=tr.rng, n_in=net.hidden_vid, 
    W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], n_out=net.n_class))


# function computing the number of errors
errors = layers[-1].errors(y)

# gradient descent
# parameter list
for layer in video_cnn.layers: 
    params.extend(layer.params)

# pre-trained dbn parameter last layer  (W, b) doesn't need to incorporate into the params
# for calculating the gradient

# softmax layer params
params.extend(layers[-1].params)

# cost function
cost = layers[-1].negative_log_likelihood(y)

# regularisation
# symbolic Theano variable that represents the L1 regularization term
for p in params:
    L1 += T.sum(abs(p))
    L2 += T.sum(p**2)
# cost loss
cost = cost + reg.L1_vid * L1 + reg.L2_vid * L2

# gradient list
gparams = T.grad(cost, params)


def get_update(i): return update[i]/(batch.mini/batch.micro)

for i, (param, gparam) in enumerate(zip(params, gparams)):
    # shape of the parameters
    shape = param.get_value(borrow=True).shape
    # init updates := zeros
    update.append(_shared(zeros(shape, dtype=config.floatX)))
    # micro_updates: sum of lr*grad
    micro_updates.append((update[i], update[i] + learning_rate*gparam))
    # re-init updates to zeros
    mini_updates.append((update[i], zeros(shape, dtype=config.floatX)))

    if use.mom:
        last_upd.append(_shared(zeros(shape, dtype=config.floatX)))
        v = momentum * last_upd[i] - get_update(i)
        mini_updates.append((last_upd[i], v))
        if mom.nag: # nesterov momentum
            mini_updates.append((param, param + momentum*v - get_update(i)))
        else:
            mini_updates.append((param, param + v))
    else:    
        mini_updates.append((param, param - get_update(i)))

####################################################################
####################################################################
print "\n%s\n\tcompiling\n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 
# compile functions
# ------------------------------------------------------------------------------
def get_batch(_data): 
    pos_mini = idx_mini*batch.mini
    idx1 = pos_mini + idx_micro*batch.micro
    idx2 = pos_mini + (idx_micro+1)*batch.micro
    return _data[idx1:idx2]

def givens(dataset_):
    return {x: get_batch(dataset_[0]),
            y: get_batch(dataset_[1]),
            x_skeleton: get_batch(dataset_[2])}

print 'compiling apply_updates'
apply_updates = function([], 
    updates=mini_updates, 
    on_unused_input='ignore')

print 'compiling train_model'
train_model = function([idx_mini, idx_micro], [cost, errors], 
    updates=micro_updates, 
    givens=givens((x_, y_int32, x_skeleton_)), 
    on_unused_input='ignore')

print 'compiling test_model'
test_model = function([idx_mini, idx_micro], [cost, errors], 
    givens=givens((x_, y_int32, x_skeleton_)),
    on_unused_input='ignore')

####################################################################
####################################################################
print "\n%s\n\ttraining\n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 
time_start = 0
best_valid = inf
# main loop
# ------------------------------------------------------------------------------
lr_decay_epoch = 0
n_lr_decays = 0
train_ce, valid_ce = [], []
flag=True
global insp_
insp_ = None

res_dir = save_results(train_ce, valid_ce, res_dir, params=params)

save_params(params, res_dir)

lr.decay = 1

for epoch in xrange(tr.n_epochs):
    ce = []
    print_params(params) 
    ####################################################################
    ####################################################################
    print "\n%s\n\t epoch %d \n%s"%('-'*30, epoch, '-'*30)
    ####################################################################
    ####################################################################
    time_start = time()
    for i in range(loader.n_iter_train):     
        #load data
        time_start_iter = time()
        loader.next_train_batch(x_, y_, x_skeleton_)
	print ('tr.batch_size_before=%d'%tr.batch_size)
        tr.batch_size = y_.get_value(borrow=True).shape[0]
	print ('tr.batch_size_after=%d'%tr.batch_size)
        ce.append(_batch(train_model, tr.batch_size, batch, True, apply_updates)[0])
        print "the %d iteration,time used:%d"%(i,time()-time_start_iter) 
        #timing_report(i, time()-time_start_iter, tr.batch_size, res_dir)
        print "\t| "+ training_report(ce[-1]) + ", finish total of: 0." + str(i*1.0/loader.n_iter_train)
    # End of Epoch
    ####################################################################
    ####################################################################
    print "\n%s\n\t End of epoch %d, \n printing some debug info.\n%s" \
        %('-'*30, epoch, '-'*30)
    ####################################################################
    ####################################################################
    # print insp_
    train_ce.append(_avg(ce))
    # validate
    valid_ce.append(test_lio_skel(use, test_model, batch, drop, tr.rng, epoch, tr.batch_size, x_, y_, loader, x_skeleton_))

    # save best params
    # if valid_ce[-1][1] < 0.25:
    res_dir = save_results(train_ce, valid_ce, res_dir, params=params)
    if not tr.moved: res_dir = move_results(res_dir)

    if valid_ce[-1][1] < best_valid:
        save_params(params, res_dir, "best")
    save_params(params, res_dir)

    if valid_ce[-1][1] < best_valid:
        best_valid = valid_ce[-1][1]

    # epoch report
    epoch_report(epoch, best_valid, time()-time_start, time()-prog_start_time, learning_rate.get_value(borrow=True),\
        train_ce[-1], valid_ce[-1], res_dir)
    # make_plot(train_ce, valid_ce)

    if lr.decay_each_epoch:
        learning_rate.set_value(float32(learning_rate.get_value(borrow=True)*lr.decay))
    # elif lr.decay_if_plateau:
    #     if epoch - lr_decay_epoch > tr.patience \
    #         and valid_ce[-1-tr.patience][1] <= valid_ce[-1][1]:

    #         write("Learning rate decay: validation error stopped improving")
    #         lr_decay_epoch = epoch
    #         n_lr_decays +=1
    #         learning_rate.set_value(float32(learning_rate.get_value(borrow=True)*lr.decay_big))
    # if epoch == 0: 
        # learning_rate.set_value(float32(3e-4))
    # else:
        # learning_rate.set_value(float32(learning_rate.get_value(borrow=True)*lr.decay))
    loader.shuffle_train()
    print"whole programe time used:%3.3f"%((time()-prog_start_time)/3600.)




