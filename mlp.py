import theano
import theano.tensor as T
import numpy as np

from layer import LogitLayer, Layer
from basenet import SuperVisedBaseNet

class MLP(SuperVisedBaseNet):
    """Multilayer Perceptron

    Parameters
    ----------
    n_in : int
        dimension of the input to the layer
    
    n_hiddens : list[int]
        dimensions of the input to hidden layers

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

    activation : function, optional (default=T.tanh)
        activation function
    
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