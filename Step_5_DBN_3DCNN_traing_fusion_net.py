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
from numpy import zeros, empty, inf, float32, random

# theano imports
from theano import function, config, shared
import theano.tensor as T

# customized imports
from dbn.GRBM_DBN import GRBM_DBN
from conv3d_chalearn import conv3d_chalearn
from convnet3d import LogRegr, HiddenLayer, DropoutLayer

#  modular imports
# the hyperparameter set the data dir, use etc classes, it's important to modify it according to your need
from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop,\
                                     net,  DataLoader_with_skeleton_normalisation
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
pc = "wudi"
if pc=="wudi":
    src = r"/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/hdf5Dest_140/"
    res_dir_ = r"/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/result_temp/multi/"# dir of original data -- note that wudi has decompressed it!!!
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
if False:
    import theano 
    theano.config.compute_test_value = 'warn' #debug mode
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

# in shape: #frames * gray/depth * body/hand * 4 maps
import cPickle
f = open('SK_normalization.pkl','rb')
SK_normalization = cPickle.load(f)
Mean1 = SK_normalization ['Mean1']
Std1 = SK_normalization['Std1']


f = open('CNN_normalization.pkl','rb')
CNN_normalization = cPickle.load(f)
Mean_CNN = CNN_normalization ['Mean_CNN']
Std_CNN = CNN_normalization['Std_CNN']


# customized data loader for both video module and skeleton module
loader = DataLoader_with_skeleton_normalisation(src, tr.batch_size, Mean_CNN, Std_CNN, Mean1, Std1) # Lio changed it to read from HDF5 files

####################################################################
# DBN for skeleton modules
#################################################################### 
# ------------------------------------------------------------------------------
# symbolic variables
x_skeleton = ndtensor(len(tr._skeleon_in_shape))(name = 'x_skeleton') # video input
x_skeleton_ = _shared(empty(tr._skeleon_in_shape))

dbn = GRBM_DBN(numpy_rng=random.RandomState(123), n_ins=891, \
                hidden_layers_sizes=[2000, 2000, 1000], n_outs=101, input_x=x_skeleton, label=y )  
# we load the pretrained DBN skeleton parameteres here
dbn.load_params_DBN('/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/result_temp/dbn/try/57.6% 2018.05.06.23.42.32/paramsbest.zip')


####################################################################
# 3DCNN for video module
#################################################################### 
# we load the CNN parameteres here
use.load = True
load_path = '/home/zhiquan/fancy/meterials/chalearn2014_fancy_data/result_temp/3dcnn/try/55.0% 2018.05.07.21.06.05/'
video_cnn = conv3d_chalearn(x, use, lr, batch, net, reg, drop, mom, tr, res_dir, load_path)

#####################################################################
# fuse the ConvNet output with skeleton output  -- need to change here
######################################################################  
out = T.concatenate([video_cnn.out, dbn.sigmoid_layers[-1].output], axis=1)

# some activation inspection
insp =  []
for insp_temp in video_cnn.insp_mean:    insp.append(insp_temp)
for layer in dbn.sigmoid_layers:    insp.append(T.mean(layer.output))

# ------------------------------------------------------------------------------
#MLP layer                
layers.append(HiddenLayer(out, n_in=net.hidden, n_out=net.hidden, rng=tr.rng, 
    W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], activation=net.activation))
out = layers[-1].output

if tr.inspect: insp.append( T.mean(out))
if use.drop: out = DropoutLayer(out, rng=tr.rng, p=drop.p_hidden).output

insp = T.stack(insp)
       
# softmax layer
layers.append(LogRegr(out, rng=tr.rng, n_in=net.hidden, 
    W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], n_out=net.n_class))
# number of inputs for MLP = (# maps last stage)*(# convnets)*(resulting video shape) + trajectory size
print 'MLP:', video_cnn.n_in_MLP, "->", net.hidden_penultimate, "+", net.hidden_traj, '->', \
   net.hidden, '->', net.hidden, '->', net.n_class, ""

# cost function
cost = layers[-1].negative_log_likelihood(y)

# function computing the number of errors
errors = layers[-1].errors(y)

# gradient descent
# parameter list
for layer in video_cnn.layers: 
    params.extend(layer.params)

# pre-trained dbn parameter last layer  (W, b) doesn't need to incorporate into the params
# for calculating the gradient
print 'len of dbn.params:%d'%len(dbn.params)
params.extend(dbn.params[:-2])

# MLP hidden layer params
params.extend(layers[-2].params)
# softmax layer params
params.extend(layers[-1].params)

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
if True:
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
        tr.batch_size = y_.get_value(borrow=True).shape[0]
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
    epoch_report(epoch, best_valid, time()-time_start, learning_rate.get_value(borrow=True),\
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



