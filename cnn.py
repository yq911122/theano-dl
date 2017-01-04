import theano
import theano.tensor as T
import numpy as np

from layer import LogitLayer, Layer, ConvPoolLayer
from basenet import SuperVisedBaseNet

class CNN2(SuperVisedBaseNet):
    """docstring for CNN2"""
    def __init__(self, img_shape, n_feats, poolsizes, filter_shapes,  n_hidden, n_out, rng=None, activations=None, batch_size=50, learning_rate=0.1, n_epoch=20, criterion=0.05, penalty='l1', alpha=0.001):
        super(CNN2, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha)

        n_layers = len(poolsizes) + 2

        if activations is None:
            activations = [None]*n_layers

        height, width = img_shape
        for i in xrange(len(poolsizes)):
            filter_height, fitler_width = filter_shapes[i]
            filter_shape = (n_feats[i+1], n_feats[i], filter_height, fitler_width)

            pool_height, pool_width = poolsizes[i]

            in_shape = (batch_size, n_feats[i], height, width)

            layeri = ConvPoolLayer(
                 filter_shape=filter_shape, 
                 image_shape=in_shape, 
                 poolsize=poolsizes[i], 
                 rng=rng, 
                 activation=activations[i])

            if i == 0:
                self.addLayer(layeri, ('reshape', in_shape))
            else:
                self.addLayer(layeri)
            
            height = (height - filter_height + 1) / pool_height
            width = (width - fitler_width + 1) / pool_width

        hidden_layer = Layer(n_in=n_feats[-1] * height * width, n_out=n_hidden, activation=activations[-2])

        self.addLayer(hidden_layer, ('flatten', 2))

        logit_layer = LogitLayer(n_in=n_hidden, n_out=n_out)
        self.addLayer(logit_layer)