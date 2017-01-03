import theano
import theano.tensor as T
import numpy as np

from grad import MSGD
from layer import LogitLayer, Layer, ConvPoolLayer 

class BaseNet(object):
    """docstring for BaseNet"""
    def __init__(self):
        super(BaseNet, self).__init__()
        self.model = MSGD()

    def fit(self, train_X, train_y, valid_X, valid_y):
        self.model.fit(train_X, train_y, valid_X, valid_y)
        return self

    def predict(self, test_X):
        return self.model.predict(test_X)

    def predict_proba(self, test_X):
        return self.model.predict_proba(test_X)

    def score(self, test_X, test_y):
        return self.model.score(test_X, test_y)

class MLP(BaseNet):
    """docstring for MLP"""
    def __init__(self, n_in, n_hidden, n_out, rng=None, activition=T.tanh):
        super(MLP, self).__init__()
        hidden_layer = Layer(n_in, n_hidden, rng, activation=activition)
        logit_layer = LogitLayer(n_hidden, n_out, rng)
        self.model.addLayer(hidden_layer)
        self.model.addLayer(logit_layer)

class CNN2(BaseNet):
    """docstring for CNN2"""
    def __init__(self, img_shape, n_feats, poolsizes, filter_shapes,  n_hidden, n_out, batch_size=None, rng=None, activations=None):
        super(CNN2, self).__init__()
        n_layers = len(poolsizes) + 2

        if activations is None:
            activations = [None]*n_layers

        if batch_size is None:
            batch_size = self.model.batch_size
        else: self.model.batch_size = batch_size

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
                self.model.addLayer(layeri, ('reshape', in_shape))
            else:
                self.model.addLayer(layeri)
            
            height = (height - filter_height + 1) / pool_height
            width = (width - fitler_width + 1) / pool_width

        hidden_layer = Layer(n_in=n_feats[-1] * height * width, n_out=n_hidden, activation=activations[-2])

        self.model.addLayer(hidden_layer, ('flatten', 2))

        logit_layer = LogitLayer(n_in=n_hidden, n_out=n_out)
        self.model.addLayer(logit_layer)


from test import load_data

url = './data/mnist.pkl.gz'

datasets = load_data(url)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# model = MLP(28 * 28, 500, 10)
# model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)

img_shape = (28, 28)
n_feats = (1, 20, 50)
poolsizes = ((2,2),(2,2))
filter_shapes = ((5,5),(5,5))
n_hidden = 500
n_out = 10
activations=[T.tanh]*2

model = CNN2(
    img_shape=img_shape,
    n_feats=n_feats,
    poolsizes=poolsizes,
    filter_shapes=filter_shapes,
    n_hidden=n_hidden,
    n_out=n_out)

model.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)