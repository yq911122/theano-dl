import theano
import theano.tensor as T
import numpy as np

from layer import dA
from basenet import SuperVisedBaseNet, UnSuperVisedBaseNet
from mlp import MLP

class SdA(object):
    """Stacked denoising Autoencoders

    Parameters
    ----------
    n_in : int
        dimension of the input to the layer
    
    n_hiddens : list[int]
        dimensions of the input to hidden layers

    n_out : int
        dimension of the output

    corruption_level : List[float]
        corruption level of input to each layer

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
    
    Attributes
    ----------
    fine_model : object; MLP
        final fit and predict model

    prefit_model : object; UnSuperVisedBaseNet
        pre fit model

    """
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

    def fit(self, train_X, train_y, valid_X, valid_y):
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