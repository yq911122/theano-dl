import theano
import theano.tensor as T
import numpy as np

from layer import RBMLayer
from basenet import SuperVisedBaseNet, UnSuperVisedBaseNet
from mlp import MLP

class UnSuperVisedDBN(UnSuperVisedBaseNet):
    """Class for un-supervised DBN

    Parameters
    ----------
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

    greedy_fitting : boolean; optional (default=True)
        if True, the fitting will be layer-wise
    
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

    greedy_fitting : boolean; optional (default=True)
        if True, the fitting will be layer-wise
    """
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
        """initiate fitting for layer. This occurs when greedy_fitting = True. In that case, fitting will be applied layer-wise, where cost, params, error (if any) and updates (if any) will be sent to MSGD.
        
        Parameters
        ----------
        X : Theano variable, array-like
            The input variable

        layer : object
            The layer to be initated
        """
        cost, updates = layer.get_cost_updates(X, self.model.learning_rate)

        self.model.init_fitting(
            cost=cost,
            params=layer.params,
            updates=updates)

class DBN(object):
    """Deep Belief Network

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

    theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams, optional (default=None)
        random state for corruption calculation    

    batch_size : int; optional (default=50)
        batch size of input for mini-batch stochastic gradient descent (MSGD) algorithm
    
    learning_rate : float; optional (default=0.01)
        learning rate for updating parameters in MSGD. For more details, see grad.py

    n_epoch : int; optional (default=20)
        maximum epoches in training. For more details, see grad.py

    n_epoch_prefit : int; optional (default=5)
        epoches in pre-training.

    criterion : float; optional (default=0.05)
        when a validation set is provided in fitting phase, a patience mechanism will be introduced and it will update its best result only if the current result is better than ( 1 - criteron ) * previous_best_result. For more details, see grad.py

    penalty : string; optional (default='l1')
        specifying reguralization method. Currently only either l1 or l2 is supported. If l1, penalty = sum(abs(W)); If l2, penalty = sum(W ** 2), where W is all weight parameters. Notice that bias parameters won't be penalized.

    alpha : float; optional (default=0.001)
        specifying level of penalty to be adapted. A hyper-param for tuning model

    activation : function, optional (default=T.tanh)
        activation function

    k : int, steps for CD-K sampling
    
    Attributes
    ----------
    fine_model : object; MLP
        final fit and predict model

    prefit_model : object; UnSuperVisedDBN
        pre fit model

    """
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
        """pre-fit the model given train_X as input

        Parameters
        ----------
        train_X : Theano shared variable, shape = [n_sample, n_features]
            The training set; n_features should be consistent with the nodes in the first layer after input transformation

        Return
        ------
        self : object
        """
        self.prefit_model.fit(train_X)
        return self

    def fit(self, train_X, train_y, valid_X=None, valid_y=None):
        """fit the network. If validation set is provided, then a patience mechanism will be apdated in gradient decent process

        Parameters
        ----------
        train_X : Theano shared variable, shape = [n_sample, n_features]
            The training set; n_features should be consistent with the nodes in the first layer after input transformation

        train_y : Theano shared variable, shape = [n_sample]
            The target of the training set
        
        valid_X : Theano shared variable, shape = [n_val_sample, n_features], optional (default=None)
            The validation set; n_features should be consistent with the nodes in the first layer after input transformation

        valid_y : Theano shared variable, shape = [n_val_sample], optional (default=None)
            The target of the validation set

        Return
        ------
        self : object

        """
        self.fine_model.fit(train_X, train_y, valid_X, valid_y)
        return self

    def predict(self, test_X):
        """predict class for test_X

        Parameters
        ----------
        test_X : Theano shared variable, array-like, shape = [n_sample, n_feature]

        Return
        ------
        pred_y : array-like, shape = [n_sample]
            The predict value of input

        """
        return self.fine_model.predict(test_X)

    def predict_proba(self, test_X):
        """predict probability of class for test_X

        Parameters
        ----------
        test_X : Theano shared variable, array-like, shape = [n_sample, n_feature]

        Return
        ------
        proba : array-like, shape = [n_sample, n_class]
            The class probabilities of the input samples

        """ 
        return self.fine_model.predict_proba(test_X)