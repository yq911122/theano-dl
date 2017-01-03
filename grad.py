# from __future__ import print_function
import six.moves.cPickle as pickle

import theano
import theano.tensor as T
import numpy as np

import timeit

class MSGD(object):
    """docstring for MSGD"""
    _support_penalties = {'l1', 'l2'}

    def __init__(self, batch_size=50, learning_rate=0.1, n_epoch=1000, criterion=0.05, penalty='l1', alpha=0.001):
        super(MSGD, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.criterion = criterion

        self.cost = None
        self.errors = None

        self.alpha = alpha
        self.penalty = penalty

        self.index = T.lscalar()

        self.layers = []
        self.params = []
        self.x_trans_params = []

    def addLayer(self, layer, input_transform_parmas=None):
        self.layers.append(layer)
        self.params += layer.params
        self.x_trans_params.append(input_transform_parmas)

    def init_fitting(self, X, y):

        for layer, tran in zip(self.layers, self.x_trans_params):
            if tran is not None:
                _name, _par = tran
                try:
                    method = getattr(X, _name)
                    X = method(_par)
                except AttributeError:
                    raise AttributeError("No such method exists.")
            X = layer.cal_output(X)


        W = [param for param in self.params if param.ndim > 1]

        if self.penalty == 'l1':
            w_sum = [T.sum(abs(wi)) for wi in W]
        else:
            w_sum = [T.sum(wi**2) for wi in W]

        penalty = T.sum(w_sum)

        self.cost = self.layers[-1].cost(y) + self.alpha*penalty
        self.error = self.layers[-1].error
        return self

    def fit(self, train_X, train_y, valid_X, valid_y):
        x = T.matrix('x')
        y = T.ivector('y')

        self.init_fitting(x, y)

        batch_size, learning_rate, index, params = self.batch_size, self.learning_rate, self.index, self.params
        n_train_batches = train_X.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_X.get_value(borrow=True).shape[0] / batch_size

        grads = [T.grad(self.cost, param) for param in params]
        updates = [(params_i, params_i - learning_rate*grad_i) for params_i, grad_i in zip(params, grads)]        

        train_model = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=updates,
            givens={
                x: train_X[index*batch_size: (index+1)*batch_size],
                y: train_y[index*batch_size: (index+1)*batch_size]
            })

        validate_model = theano.function(
            inputs=[index],
            outputs=self.error(y),
            givens={
                x: valid_X[index*batch_size: (index+1)*batch_size],
                y: valid_y[index*batch_size: (index+1)*batch_size]
            })

        print('... training the model')
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                                      # found
        improvement_threshold = 1 - self.criterion  # a relative improvement of this much is
                                      # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        while (epoch < self.n_epoch) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                         for i in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss

                        params = self.params

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
            )
            % (best_validation_loss * 100.)
        )
        print('The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time)))

        self.params = params
        # save the best model
        with open('best_model_params.pkl', 'wb') as f:
            pickle.dump(params, f)

        self.update_layer_params()
        return self

    def update_layer_params(self):
        n_params = sum(len(layer.params) for layer in self.layers)
        if n_params != len(self.params):
            raise ValueError("Inconsistent parameters number")

        j = 0
        for layer in self.layers:
            for i in xrange(len(layer.params)):
                layer.params[i] = self.params[j]
                j += 1

    def predict(self, test_X):
        index = self.index
        n_size = test_X.get_value(borrow=True).shape[0]
        # n_size = 50
        x = T.matrix('x')
        output = x
        for layer in self.layers:
            output = layer.cal_output(output)
        pred = theano.function(
            inputs=[index],
            outputs=self.layers[-1].get_predict(),
            givens={x: test_X[:index]}
            )
        return pred(n_size)

    def predict_proba(self, test_X):
        index = self.index
        n_size = test_X.get_value(borrow=True).shape[0]
        # n_size = 10
        x = T.matrix('x')
        output = x
        for layer in self.layers:
            output = layer.cal_output(output)
        pred = theano.function(
            inputs=[index],
            outputs=output,
            givens={x: test_X[:index]}
            )
        return pred(n_size)

    def score(self, test_X, test_y):
        x = T.matrix('x')
        y = T.ivector('y')
        index = self.index
        n_size = test_X.get_value(borrow=True).shape[0]
        output = x
        for layer in self.layers:
            output = layer.cal_output(output)
        pred_y = self.layers[-1].get_predict()
        error = T.mean(T.neq(pred_y, y))
        get_error = theano.function(
            inputs=[index],
            outputs=error,
            givens={x: test_X[:index], y: test_y[:index]}
            )
        return get_error(n_size)


class UnSupervisedMSGD(object):
    """docstring for UnSupervisedMSGD"""
    def __init__(self, arg):
        super(UnSupervisedMSGD, self).__init__()
        self.arg = arg
        