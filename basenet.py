import theano
import theano.tensor as T
import numpy as np

from grad import MSGD

class BaseNet(object):
    """docstring for BaseNet"""
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

        self.cost = None
        self.error = None

    def _cal_penalty(self):
        W = [param for param in self.params if param.ndim > 1]

        if self.penalty == 'l1':
            w_sum = [T.sum(abs(wi)) for wi in W]
        else:
            w_sum = [T.sum(wi**2) for wi in W]

        return T.sum(w_sum)

    def _update_output(self, X):
        for layer, tran in zip(self.layers, self.x_trans_params):
            if tran is not None:
                _name, _par = tran
                try:
                    method = getattr(X, _name)
                    X = method(_par)
                except AttributeError:
                    raise AttributeError("No such method exists.")
            X = layer.cal_output(X)

    def addLayer(self, layer, input_transform_parmas=None):
        self.layers.append(layer)
        self.params += layer.params
        self.x_trans_params.append(input_transform_parmas)

class SuperVisedBaseNet(BaseNet):
    """docstring for SuperVisedBaseNet"""
    def __init__(self, batch_size=50, learning_rate=0.1, n_epoch=20, criterion=0.05, penalty='l1', alpha=0.001):
        super(SuperVisedBaseNet, self).__init__(
            batch_size=batch_size,
            learning_rate=learning_rate, 
            n_epoch=n_epoch,
            criterion=criterion,
            penalty=penalty, 
            alpha=alpha)

    def init_fitting(self, X, y):
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
    """docstring for UnSuperVisedBaseNet"""
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
        error = layer.error(X) if hasattr(layer, 'error') else None
        updates = layer.updates if hasattr(layer, 'updates') else None

        self.model.init_fitting(
            cost=layer.cost(X),
            params=layer.params,
            error=error, 
            updates=updates)

    def fit(self, train_X, valid_X=None):
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
        index = T.lscalar()
        n_size = data.get_value(borrow=True).shape[0]
        predict = theano.function(
                inputs=[index],
                outputs=layer.cal_output(var),
                givens={var: data[:index]})
        return theano.shared(np.asarray(predict(n_size), dtype=theano.config.floatX), borrow=True)