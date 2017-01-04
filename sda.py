import theano
import theano.tensor as T
import numpy as np

from layer import dA
from basenet import SuperVisedBaseNet, UnSuperVisedBaseNet
from mlp import MLP

class SdA(object):
    """docstring for SdA"""
    def __init__(self, n_in, n_hiddens, n_out, corruption_levels, rng=None, theano_rng=None, batch_size=50, learning_rate=0.1, n_epoch=20, n_epoch_prefit=5, criterion=0.05, penalty='l1', alpha=0.001):
        super (SdA, self).__init__()

        self.fine_model = MLP(
            n_in=n_in, 
            n_hiddens=n_hiddens, 
            n_out=n_out,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epoch=n_epoch, 
            criterion=criterion, 
            penalty=penalty, 
            alpha=alpha)

        self.prefit_model = UnSuperVisedBaseNet(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch_prefit,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha,
            greedy_fitting=True)

        i = 0
        for layer in self.fine_model.layers[:-1]:
            input_size = n_in if i == 0 else n_hiddens[i-1]

            dA_layer = dA(
                n_in=input_size,
                n_hidden=n_hiddens[i],
                rng=rng,
                theano_rng=theano_rng,
                W=layer.W,
                bhid=layer.b,
                corruption_level=corruption_levels[i])

            self.prefit_model.addLayer(dA_layer)
            i += 1

    def prefit(self, train_X):
        self.prefit_model.fit(train_X)
        return self

    def fit(self, train_X, train_y, valid_X, valid_y):
        self.fine_model.fit(train_X, train_y, valid_X, valid_y)
        return self