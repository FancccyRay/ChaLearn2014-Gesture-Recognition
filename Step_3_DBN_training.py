"""
Video classifier using a 3D deep convolutional neural network
and DBN, fusing the two result together
Data: ChaLearn 2014 gesture challenge: gesture recognition
original code by: Lionel Pigou
Code modulated by: Di Wu   stevenwudi@gmail.com
2015-06-12
"""
# various imports
from cPickle import load
from glob import glob
from time import time, localtime
from gzip import GzipFile
import os
# numpy imports
from numpy import zeros, empty, inf, float32, random, linspace
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
                                     net,  DataLoader_with_skeleton_normalisation
from functions.train_functions import _shared, _avg, write, ndtensor, print_params, lin,\
                                      training_report, epoch_report, _batch,\
                                      save_results, move_results, save_params, test_lio_skel

prog_start_time = time()
print"programe start time:%d"%prog_start_time
####################################################################
####################################################################
print "\n%s\n\t initializing \n%s"%(('-'*30,)*2)
####################################################################
####################################################################
# source and result directory
#pc = "wudi"
pc = "wudi_linux"
if pc=="wudi":
    src = r"D:\Chalearn2014\Data_processed"
    res_dir_ = r"D:\Chalearn2014\result"# dir of original data -- note that wudi has decompressed it!!!
elif pc == "wudi_linux":
    src = r"/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/hdf5Dest_140/"
    res_dir_ = r"/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/result_temp/dbn/"# dir of original data -- note that wudi has decompressed it!!!
elif pc=="lio":
    src = "/mnt/wd/chalearn/preproc"
    res_dir_ = "/home/lpigou/chalearn_wudi/try"


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


# load the skeleton normalisation --Lio didn't normalise video input, but should we?
import cPickle
f = open('SK_normalization.pkl','rb')
SK_normalization = cPickle.load(f)
Mean1 = SK_normalization ['Mean1']
Std1 = SK_normalization['Std1']

# customized data loader for both video module and skeleton module
#loader = DataLoader_with_skeleton(src, tr.batch_size, Mean1, Std1) # Lio changed it to read from HDF5 files
loader = DataLoader_with_skeleton_normalisation(src, tr.batch_size, 0, 1, Mean1, Std1) # Lio changed it to read from HDF5 files
####################################################################
# DBN for skeleton modules
#################################################################### 
# ------------------------------------------------------------------------------
# symbolic variables
x_skeleton = ndtensor(len(tr._skeleon_in_shape))(name = 'x_skeleton') # video input
x_skeleton_ = _shared(empty(tr._skeleon_in_shape))
########sample number:39,hidden_layer_size=[2000,1000]->error rate=77.1%;[2000,2000,1000],78.4%

dbn = GRBM_DBN(numpy_rng=random.RandomState(123), n_ins=891, \
                hidden_layers_sizes=[2000,2000,1000], n_outs=101, input_x=x_skeleton, label=y )  
# we load the pretrained DBN skeleton parameteres here, currently pretraining is done
# unsupervisedly, we can load the supervised pretrainining parameters later
#                
dbn.load_params_DBN("/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/result_temp/dbn/try/63.9% 2018.05.06.19.54.43/paramsbest.zip")  


cost = dbn.finetune_cost

# function computing the number of errors
errors = dbn.errors


# wudi add the mean and standard deviation of the activation values to exam the neural net
# Reference: Understanding the difficulty of training deep feedforward neural networks, Xavier Glorot, Yoshua Bengio
out_mean = T.stack(dbn.out_mean)
out_std = T.stack(dbn.out_std)


gparams = T.grad(cost, dbn.params)
params = dbn.params

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
train_model = function([idx_mini, idx_micro], [cost, errors, out_mean, out_std], 
    updates=micro_updates, 
    givens=givens((x_, y_int32, x_skeleton_)),
    on_unused_input='ignore')

print 'compiling test_model'
test_model = function([idx_mini, idx_micro], [cost, errors, out_mean, out_std], 
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
out_mean_all, out_std_all = [], []
flag=True
global insp_
insp_ = None

res_dir = save_results(train_ce, valid_ce, res_dir, params=params)

save_params(params, res_dir)

# Wudi makes thie to explicity control the learning rate

learning_rate_map = linspace(lr.start, lr.stop, tr.n_epochs)


for epoch in xrange(tr.n_epochs):
    ce = []
    out_mean_train = []
    out_std_train = []
    print_params(params) 
    ####################################################################
    ####################################################################
    print "\n%s\n\t epoch %d \n%s"%('-'*30, epoch, '-'*30)
    ####################################################################
    ####################################################################
    time_start = time()
    print loader.n_iter_train
    for i in range(loader.n_iter_train):     
        #load data
        time_start_iter = time()
        loader.next_train_batch(x_, y_, x_skeleton_)
        #tr.batch_size = y_.get_value(borrow=True).shape[0]
        ce_temp, out_mean_temp, out_std_temp = _batch(train_model, tr.batch_size, batch, True, apply_updates)
	#print out_mean_train, out_std_train
        ce.append(ce_temp)
        out_mean_train.append(out_mean_temp)
        out_std_train.append(out_std_temp)

        print "Training: No.%d iter of Total %d, %d s"% (i,loader.n_iter_train, time()-time_start_iter)  \
                + "\t| negative_log_likelihood "+ training_report(ce[-1]) 
    # End of Epoch
    ####################################################################
    ####################################################################
    print "\n%s\n\t End of epoch %d, \n printing some debug info.\n%s" \
        %('-'*30, epoch, '-'*30)
    ####################################################################
    ####################################################################
    print ce
    train_ce.append(_avg(ce))
    out_mean_all.append(_avg(out_mean_train))
    out_std_all.append(_avg(out_std_train))
    # validate
    valid_ce.append(test_lio_skel(use, test_model, batch, drop, tr.rng, epoch, tr.batch_size, x_, y_, loader, x_skeleton_))

    # save best params
    res_dir = save_results(train_ce, valid_ce, res_dir, params=params, out_mean_train=out_mean_all,out_std_train=out_std_all)
    if not tr.moved: res_dir = move_results(res_dir)

    if valid_ce[-1][1] < best_valid:
        save_params(params, res_dir, "best")
    save_params(params, res_dir)

    if valid_ce[-1][1] < best_valid:
        best_valid = valid_ce[-1][1]

    # epoch report
    epoch_report(epoch, best_valid, time()-time_start, learning_rate.get_value(borrow=True),\
        train_ce[-1], valid_ce[-1], res_dir)

    # decay the learning rate
    learning_rate.set_value(float32(learning_rate_map[epoch]))
    loader.shuffle_train()
    print 'whole programe time used: %3.3f h'%((time()-prog_start_time)/ 3600.)



