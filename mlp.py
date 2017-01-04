import theano
import theano.tensor as T
import numpy as np

from layer import LogitLayer, Layer
from basenet import SuperVisedBaseNet

class MLP(SuperVisedBaseNet):
    """docstring for MLP"""
    def __init__(self, n_in, n_hiddens, n_out, rng=None, activation=T.tanh, batch_size=50, learning_rate=0.1, n_epoch=20, criterion=0.05, penalty='l1', alpha=0.001):
        super(MLP, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha)

        n_hidden_layers = len(n_hiddens)

        for i in xrange(n_hidden_layers):
            input_size = n_in if i == 0 else n_hiddens[i-1]
            hidden_layer = Layer(input_size, n_hiddens[i], rng, activation=activation)
            self.addLayer(hidden_layer)

        logit_layer = LogitLayer(n_hiddens[-1], n_out, rng)
        self.addLayer(logit_layer)