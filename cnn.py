import theano
import theano.tensor as T
import numpy as np

from layer import LogitLayer, Layer, ConvPoolLayer2D
from basenet import SuperVisedBaseNet

class CNN2(SuperVisedBaseNet):
    """2D Convolutional Network. It has the structure of: ConvPoolLayer2D -> ... ConvPoolLayer2D -> Layer -> LogitLayer

    Parameters
    ----------
    img_shape : tuple (batch_size, n_input_feature_maps, input_height, input_width)

    n_feats : list[int]; number of feature maps of layers

    poolsizes : list[tuple()]; size of pooling of layers

    filter_shapes : list[tuple()]; filter shapes of layers
    
    n_hidden : int
        dimension of the input to the hidden layer

    n_out : int
        dimension of the output
    
    rng : np.random.RandomState, optional (default=None)
        random state for parameter random initation
        
    batch_size : int; optional (default=50)
        batch size of input for mini-batch stochastic gradient descent (MSGD) algorithm
    
    learning_rate : float; optional (default=0.01)
        learning rate for updating parameters in MSGD. For more details, see grad.py

    n_epoch : int; optional (default=20)
        maximum epoches in training. For more details, see grad.py

    criterion : float; optional (default=0.05)
        when a validation set is provided in fitting phase, a patience mechanism will be introduced and it will update its best result only if the current result is better than ( 1 - criteron ) * previous_best_result. For more details, see grad.py

    penalty : string; optional (default='l1')
        specifying reguralization method. Currently only either l1 or l2 is supported. If l1, penalty = sum(abs(W)); If l2, penalty = sum(W ** 2), where W is all weight parameters. Notice that bias parameters won't be penalized.

    alpha : float; optional (default=0.001)
        specifying level of penalty to be adapted. A hyper-param for tuning model

    activations : list[function], optional (default=None)
        activation functions of layers (except the output LogitLayer)
    
    Attributes
    ----------
    model : object;
        MSGD model. For more details, see grad.py

    params : list;
        Flattened list that stores all parameters in the network, where the order is layer-wised: 
        [input_layer.params, hidden_layers.params, output_layer.params]

    layers : list;
        List that stores all layers in the network, like [input_layer, hidden_layers, output_layer]

    x_trans_params : list;
        List that stores input transorm of the corresponding layer, with each element as (transform_name, params). e.g., 1st element specifies the required transform from the input of the network when fitting to the input of the 1st layer

    alpha : float;
        specifying level of penalty to be adapted. A hyper-param for tuning model

    penalty : string;
        ways of pentaly
    """
    def __init__(self, 
        img_shape, 
        n_feats, 
        poolsizes, 
        filter_shapes,  
        n_hidden, 
        n_out, 
        rng=None, 
        activations=None, 
        batch_size=50, 
        learning_rate=0.1, 
        n_epoch=20, 
        criterion=0.05, 
        penalty='l1', 
        alpha=0.001):
        super(CNN2, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha)

        n_layers = len(poolsizes) + 1

        if activations is None:
            activations = [T.nnet.sigmoid]*n_layers

        height, width = img_shape
        for i in xrange(len(poolsizes)):
            filter_height, fitler_width = filter_shapes[i]
            filter_shape = (n_feats[i+1], n_feats[i], filter_height, fitler_width)

            pool_height, pool_width = poolsizes[i]

            in_shape = (batch_size, n_feats[i], height, width)

            layeri = ConvPoolLayer2D(
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