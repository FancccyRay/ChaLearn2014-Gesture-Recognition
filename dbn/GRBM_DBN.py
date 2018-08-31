"""
"""
import cPickle
import gzip
import os
import sys
import time

import numpy
from numpy import float32, random, floor
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM
from grbm import GBRBM
from utils import zero_mean_unit_variance
from utils import normalize

import warnings
warnings.filterwarnings("ignore")

class GRBM_DBN(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10, finetune_lr=0.1, input_x=None, label=None):

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        # wudi add the mean and standard deviation of the activation values to exam the neural net
        # Reference: Understanding the difficulty of training deep feedforward neural networks, Xavier Glorot, Yoshua Bengio
        self.out_mean = []
        self.out_std = []

        assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        
            
        if input_x is None:
            self.x = T.matrix('x')  # the data is presented as rasterized images
        else: 
            self.x = input_x
        if label is None:
            self.y = T.ivector('y')  # the labels are presented as 1D vector
                                     # of [int] labels
        else:
            self.y = label

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.out_mean.append(T.mean(sigmoid_layer.output))
            self.out_std.append(T.std(sigmoid_layer.output))

            self.params.extend(sigmoid_layer.params)
            # Construct an RBM that shared weights with this layer
            if i == 0:
                rbm_layer = GBRBM(input=layer_input, n_in=input_size, n_hidden=hidden_layers_sizes[i], \
                W=None, hbias=None, vbias=None, numpy_rng=None, transpose=False, activation=T.nnet.sigmoid,
                theano_rng=None, name='grbm', W_r=None, dropout=0, dropconnect=0)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

        #################################################
        # Wudi change the annealing learning rate:
        #################################################
        self.state_learning_rate =  theano.shared(numpy.asarray(finetune_lr,
                                               dtype=theano.config.floatX),
                                               borrow=True)

    def pretraining_functions(self, train_set_x, batch_size, k):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for i, rbm in enumerate(self.rbm_layers):

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)
            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                    train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, annealing_learning_rate=0.999):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set
        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]


        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size


        index = T.lscalar('index')  # index to a [mini]batch
        

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        #################################################
        # Wudi change the annealing learning rate:
        #################################################
        updates.append((self.state_learning_rate, annealing_learning_rate * self.state_learning_rate))

        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * self.state_learning_rate.get_value()))



        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})


        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]


        return train_fn, valid_score

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))
        
    def load_params_DBN(self, load_file=""):
        import os
        from gzip import GzipFile
        from cPickle import dump, load
        if os.path.isfile(load_file):
            file = GzipFile(load_file, "rb")
        param_load = load(file)
        file.close()
        load_params_pos = 0
        for p in self.params:
            #print p.get_value().shape
            #print param_load[load_params_pos].shape
            p.set_value(param_load[load_params_pos], borrow=True)
            load_params_pos += 1 
        print "finish loading dbn parameters"


def test_GRBM_DBN(finetune_lr=0.2, pretraining_epochs=1,
             pretrain_lr=0.01, k=1, training_epochs=10,
             dataset='mnist.pkl.gz', batch_size=10, annealing_learning_rate=0.999):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    datasets = load_data_grbm(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    x_skeleton = T.matrix('x')
    dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
                hidden_layers_sizes=[1000, 1000, 1000],
                n_outs=10, finetune_lr=finetune_lr, input=x_skeleton)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)



    # The following part is to get the value for testing
    if False:
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 2D vector of [float32] labels

        dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
                    hidden_layers_sizes=[1000, 1000, 1000],
                    n_outs=10)
        dbn.load('dbn_params.npy')
        #train_fn, validate_model, test_model = dbn.build_finetune_functions(
        #            datasets=datasets, batch_size=batch_size,
        #            learning_rate=finetune_lr)


        valid_score_i = theano.function([index], dbn.errors,
                givens={dbn.x: valid_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                        dbn.y: valid_set_y[index * batch_size:
                                            (index + 1) * batch_size]})

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size

        validation_losses = [valid_score_i(i) for i in xrange(n_valid_batches)]

        validation_losses = valid_score_i()
        this_validation_loss = numpy.mean(validation_losses)



        ## get the actual softmax layer
        temp = theano.function([index],dbn.logLayer.p_y_given_x,
                    givens={dbn.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size]})
        temp_out = [temp(i) for i in xrange(n_valid_batches)]
        



    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        start_time_temp = time.clock()
        if i==0:
            # for GRBM, the The learning rate needs to be about one or 
            #two orders of magnitude smaller than when using
            #binary visible units and some of the failures reported in the 
            # literature are probably due to using a
            pretrain_lr_new = pretrain_lr*0.1 
        else:
            pretrain_lr_new = pretrain_lr
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr_new))
            end_time_temp = time.clock()
            print 'Pre-training layer %i, epoch %d, cost %f ' % (i, epoch, numpy.mean(c)) + ' ran for %d sec' % ((end_time_temp - start_time_temp) )


    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                annealing_learning_rate=annealing_learning_rate)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.999 # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                import warnings
                warnings.filterwarnings("ignore")
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter


                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    print dbn.state_learning_rate.get_value()

def load_data_grbm(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    from sklearn import preprocessing

    # Wudi made it a small set:
    train_set_feature = train_set[0][0:1000,:]
    [train_set_feature1, Mean1, Std1]  = zero_mean_unit_variance(train_set_feature)
    # Wudi added normalized data for GRBM
    #[train_set_feature2, Mean2, Var2] = zero_mean_unit_variance(train_set_feature)
    train_set_new_target = train_set[1][0:1000]
    train_set_x, train_set_y = shared_dataset( (train_set_feature1,train_set_new_target))

    valid_set_feature = valid_set[0]
    valid_set_feature = normalize(valid_set_feature, Mean1, Std1)
    valid_set_x, valid_set_y = shared_dataset((valid_set_feature,valid_set[1]))
    # test feature set
    test_set_feature = test_set[0]
    test_set_feature = normalize(test_set_feature, Mean1, Std1)
    test_set_x, test_set_y = shared_dataset((test_set_feature,test_set[1]))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    test_GRBM_DBN()
