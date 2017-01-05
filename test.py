import gzip
import os
import sys
import six.moves.cPickle as pickle

import theano
import theano.tensor as T
import numpy as np


def load_data(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############    

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

# from test import load_data

# url = './data/mnist.pkl.gz'

# datasets = load_data(url)

# train_set_x, train_set_y = datasets[0]
# valid_set_x, valid_set_y = datasets[1]
# test_set_x, test_set_y = datasets[2]

# model = MLP(28 * 28, 500, 10)
# model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)



# print model.score(test_set_x, test_set_y)
# img_shape = (28, 28)
# n_feats = (1, 20, 50)
# poolsizes = ((2,2),(2,2))
# filter_shapes = ((5,5),(5,5))
# n_hidden = 500
# n_out = 10
# activations=[T.tanh]*2

# model = CNN2(
#     img_shape=img_shape,
#     n_feats=n_feats,
#     poolsizes=poolsizes,
#     filter_shapes=filter_shapes,
#     n_hidden=n_hidden,
#     n_out=n_out)

# # model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)
# f = open('best_model_params.pkl', 'rb')
# params = pickle.load(f)
# f.close()

# model.params = params
# print model.score(test_set_x, test_set_y)


# model = SdA(
#     n_in=28 * 28,
#     n_hiddens=[1000, 1000, 1000],
#     n_out=10,
#     corruption_levels=[0.2,0.2,0.2])

# model.prefit(train_set_x)
# model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)

# model = MLP(
#     n_in=28 * 28,
#     n_hiddens=[1000],
#     n_out=10)
# model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)

# model = DBN(
#     n_in=28 * 28, 
#     n_hiddens=[1000, 1000, 1000], 
#     n_out=10, 
#     k=2)

# model.prefit(train_set_x)
# model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)