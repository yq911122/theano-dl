import theano
import theano.tensor as T
import numpy as np

from grad import MSGD

class BaseNet(object):  
    """Base class for un-supervised neural network. Mini-batch stochastic gradient descent algorithm is applied to train the model.

    Warning: This class should not be used directly. Use derived classes
    instead.

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

    _support_penalties = {'l1', 'l2'}

    def __init__(self, batch_size=50, learning_rate=0.1, n_epoch=20, criterion=0.05, penalty='l1', alpha=0.001):
        super(BaseNet, self).__init__()
        self.model = MSGD(
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epoch=n_epoch,
            criterion=criterion)

        self.params = []
        self.layers = []
        self.x_trans_params = []

        self.alpha = alpha
        self.penalty = penalty

        # self.cost = None
        # self.error = None

    def _cal_penalty(self):
        """calculate model penatly
        
        Return
        ------
        Penatly : Theano variable, scalar
            Only l1 or l2 penatly is supported. Otherwise, a ValueError will be raised.

        """
        W = [param for param in self.params if param.ndim > 1]

        if self.penalty == 'l1':
            w_sum = [T.sum(abs(wi)) for wi in W]
        elif self.penalty == 'l2':
            w_sum = [T.sum(wi**2) for wi in W]
        else:
            raise ValueError("{0} penalty is not supported. Only l1 or l2 is supported.".format(self.penalty))

        return T.sum(w_sum)

    def _update_output(self, X):
        """calculate output layer-wise and set then as layer output. During this porcess, input transformation will be checked layer-wise. If the layer doesn't have related transformation method, an AttributeError will be raised

        Parameters
        ----------
        X : Theano variable, array-like
            The input variable

        Return
        ------
        self : object     
        """
        for layer, tran in zip(self.layers, self.x_trans_params):
            if tran is not None:
                _name, _par = tran
                try:
                    method = getattr(X, _name)
                    X = method(_par)
                except AttributeError:
                    raise AttributeError("Layer {0} doesn't have {1} method".format(layer.__class__.__name__, _name))
            X = layer.cal_output(X)

        return self

    def addLayer(self, layer, input_transform_parmas=None):
        """Add a layer to current network. Notice that the network will be built based on the order of layers added.

        Parameters
        ----------
        layer : object; layer. For more details, see layer.py

        input_transform_parmas : tuple, optional (default=None)
            Specifying input transformation, like (transform method, method params)

        """
        self.layers.append(layer)
        self.params += layer.params
        self.x_trans_params.append(input_transform_parmas)

class SuperVisedBaseNet(BaseNet):
    """Base class for supervised neural network. Mini-batch stochastic gradient descent algorithm is applied to train the model.

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
    def __init__(self, batch_size=50, learning_rate=0.1, n_epoch=20, criterion=0.05, penalty='l1', alpha=0.001):
        super(SuperVisedBaseNet, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha)

    def init_fitting(self, X, y):
        """initiate fitting. Model cost, parameters and error (if any) will be sent to MSGD for fitting initiation.
        
        Parameters
        ----------
        X : Theano variable, array-like
            The input variable

        y : Theano variable, array-like
            The target of the training set

        Return
        ------
        self : object
        """
        self._update_output(X)

        penalty = self._cal_penalty()

        # check if cost, params, exist
        error = None
        if hasattr(self.layers[-1], 'error'):
            error = self.layers[-1].error(y)
        self.model.init_fitting(
            cost=self.layers[-1].cost(y) + self.alpha*penalty, 
            params=self.params, 
            error=error)
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
        x = T.matrix('x')
        y = T.ivector('y')
        self.init_fitting(x, y)

        if valid_X is not None and valid_y is not None:
            self.model.fit_with_valid([x, y], [train_X, train_y], [valid_X, valid_y])
        else:
            self.model.fit([x, y], [train_X, train_y])
        self.params = self.model.params
        return self

    # XXX: when dealing with CNN, there exist batch size problems
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
        index = T.lscalar()
        n_size = test_X.get_value(borrow=True).shape[0]
        # n_size = 50
        x = T.matrix('x')
        self._update_output(x)
        pred = theano.function(
            inputs=[index],
            outputs=self.layers[-1].get_predict(),
            givens={x: test_X[:index]}
            )
        return pred(n_size)

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
        index = T.lscalar()
        n_size = test_X.get_value(borrow=True).shape[0]
        # n_size = 10
        x = T.matrix('x')
        self._update_output(x)
        pred = theano.function(
            inputs=[index],
            outputs=self.layers[-1].get_predict_proba(),
            givens={x: test_X[:index]}
            )
        return pred(n_size)

    def score(self, test_X, test_y):
         """calculate the training error of test_X, test_y
         
        Parameters
        ----------
        test_X : Theano shared variable, array-like, shape = [n_sample, n_feature]

        test_y : Theano shared variable, array-like, shape = [n_samples]

        Return
        ------
        error : float; mis-classification rate

        """       
        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()
        n_size = test_X.get_value(borrow=True).shape[0]
        self._update_output(x)
        pred_y = self.layers[-1].get_predict()
        error = T.mean(T.neq(pred_y, y))
        get_error = theano.function(
            inputs=[index],
            outputs=error,
            givens={x: test_X[:index], y: test_y[:index]}
            )
        return get_error(n_size)

class UnSuperVisedBaseNet(BaseNet):
    """Base class for un-supervised neural network. Mini-batch stochastic gradient descent algorithm is applied to train the model.

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
    def __init__(self, batch_size=50, learning_rate=0.1, n_epoch=20, criterion=0.05, penalty='l1', alpha=0.001, greedy_fitting=True):
        super(UnSuperVisedBaseNet, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha)

        self.greedy_fitting = greedy_fitting

    def init_fitting(self, X):
        """initiate fitting. Model cost, parameters and error (if any) will be sent to MSGD for fitting initiation.
        
        Parameters
        ----------
        X : Theano variable, array-like
            The input variable

        Return
        ------
        self : object
        """
        self._update_output(X)

        penalty = self._cal_penalty()

        # check if cost, params, exist
        error = None
        if hasattr(self.layers[-1], 'error'):
            error = self.layers[-1].error(X)
        self.model.init_fitting(
            cost=self.layers[-1].cost(X) + self.alpha*penalty, 
            params=self.params, 
            error=error)
        return self
    
    def layer_init_fitting(self, X, layer):
        """initiate fitting for layer. This occurs when greedy_fitting = True. In that case, fitting will be applied layer-wise, where cost, params, error (if any) and updates (if any) will be sent to MSGD.
        
        Parameters
        ----------
        X : Theano variable, array-like
            The input variable

        layer : object
            The layer to be initated
        """
        error = layer.error(X) if hasattr(layer, 'error') else None
        updates = layer.updates if hasattr(layer, 'updates') else None

        self.model.init_fitting(
            cost=layer.cost(X),
            params=layer.params,
            error=error, 
            updates=updates)

    def fit(self, train_X, valid_X=None):
        """fit the network. If validation set is provided, then a patience mechanism will be apdated in gradient decent process. If greedy_fitting = True, a layer-wise fitting will be applied.

        Parameters
        ----------
        train_X : Theano shared variable, shape = [n_sample, n_features]
            The training set; n_features should be consistent with the nodes in the first layer after input transformation
        
        valid_X : Theano shared variable, shape = [n_val_sample, n_features], optional (default=None)
            The validation set; n_features should be consistent with the nodes in the first layer after input transformation

        Return
        ------
        self : object

        """
        x = T.matrix('x')
        if self.greedy_fitting:
            for layer in self.layers:
                self.layer_init_fitting(x, layer)

                if valid_X is not None:
                    self.model.fit_with_valid([x], [train_X], [valid_X])
                    valid_X = self.layer_output(layer, x, valid_X)
                else: 
                    self.model.fit([x], [train_X])

                train_X = self.layer_output(layer, x, train_X)
                layer.params = self.model.params
                x = layer.cal_output(x)
        else:
            self.init_fitting(x)

            if valid_X is not None:
                self.model.fit_with_valid([x], [train_X], [valid_X])
            else:
                self.model.fit([x], [train_X])
            self.params = self.model.params
        return self

    def layer_output(self, layer, var, data):
        """calculate output of layer.

        Parameters
        ----------
        layer : object

        var : Theano variable; variable needed to calculate output of layer

        data : Theano shared variable; storing data for var

        Return
        ------
        output : Theano shared variable; output of layer

        """
        index = T.lscalar()
        n_size = data.get_value(borrow=True).shape[0]
        predict = theano.function(
                inputs=[index],
                outputs=layer.cal_output(var),
                givens={var: data[:index]})
        return theano.shared(np.asarray(predict(n_size), dtype=theano.config.floatX), borrow=True)