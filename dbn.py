import theano
import theano.tensor as T
import numpy as np

from layer import RBMLayer
from basenet import SuperVisedBaseNet, UnSuperVisedBaseNet
from mlp import MLP

class UnSuperVisedDBN(UnSuperVisedBaseNet):
    """docstring for UnSuperVisedDBN"""
    def __init__(self, 
        batch_size=50,
        learning_rate=0.1, 
        n_epoch=5,
        criterion=0.05,
        penalty='l1', 
        alpha=0.001):

        super(UnSuperVisedDBN, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha,
            greedy_fitting=True)
        

    def layer_init_fitting(self, X, layer):
        cost, updates = layer.get_cost_updates(X, self.model.learning_rate)

        self.model.init_fitting(
            cost=cost,
            params=layer.params,
            updates=updates)

class DBN(object):
    """docstring for DBN"""    
    def __init__(self, n_in, n_hiddens, n_out, rng=None, theano_rng=None, activation=T.nnet.sigmoid, k=1, batch_size=50, learning_rate=0.1, n_epoch=20, n_epoch_prefit=2, criterion=0.05, penalty='l1', alpha=0.001):
        super(DBN, self).__init__()

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

        self.prefit_model = UnSuperVisedDBN(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch_prefit,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha)

        i = 0
        for layer in self.fine_model.layers[:-1]:
            input_size = n_in if i == 0 else n_hiddens[i-1]

            rbm_layer = RBMLayer(
                n_visible=input_size,
                n_hidden=n_hiddens[i],
                rng=rng,
                theano_rng=theano_rng,
                W=layer.W,
                hbias=layer.b,
                activation=activation,
                k=k)

            self.prefit_model.addLayer(rbm_layer)
            i += 1

    def prefit(self, train_X):
        self.prefit_model.fit(train_X)
        return self

    def fit(self, train_X, train_y, valid_X, valid_y):
        self.fine_model.fit(train_X, train_y, valid_X, valid_y)
        return self

from test import load_data

url = './data/mnist.pkl.gz'

datasets = load_data(url)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
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

model = DBN(
    n_in=28 * 28, 
    n_hiddens=[1000, 1000, 1000], 
    n_out=10, 
    k=2)

model.prefit(train_set_x)
model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)
