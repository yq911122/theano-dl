# from __future__ import print_function
import six.moves.cPickle as pickle

import theano
import theano.tensor as T
import numpy as np

import timeit

class MSGD(object):
    """docstring for MSGD"""

    def __init__(self, batch_size=50, learning_rate=0.1, n_epoch=20, criterion=0.05):
        super(MSGD, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.criterion = criterion

        self.cost = None
        self.errors = None
        self.updates = None

        self.index = T.lscalar()

        self.params = []

    def init_fitting(self, cost, params, error=None, updates=None):
        self.cost = cost
        self.params = params
        if error is not None:
            self.error = error
        if updates is not None:
            self.updates = updates
        else:
            grads = [T.grad(self.cost, param) for param in params]
            self.updates = [(params_i, params_i - self.learning_rate*grad_i) for params_i, grad_i in zip(params, grads)]

    def fit(self, variables, datasets):
        # check input
        # check if cost, params, updates are set

        batch_size, learning_rate, index, params = self.batch_size, self.learning_rate, self.index, self.params

        n_train_batches = datasets[0].get_value(borrow=True).shape[0] / batch_size

        if self.updates is None:
            grads = [T.grad(self.cost, param) for param in params]
            updates = [(params_i, params_i - learning_rate*grad_i) for params_i, grad_i in zip(params, grads)]

        train_model = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                var: data[index*batch_size: (index+1)*batch_size] for var, data in zip(variables, datasets)
            })

        print('... training the model')

        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        minibatch_avg_cost = []
        for epoch in xrange(self.n_epoch):
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost += [train_model(minibatch_index)]

        print ('cost %f' % (np.mean(minibatch_avg_cost)))

        end_time = timeit.default_timer()
       
        print('The code run for %d epochs, with %f epochs/sec' % (
            epoch + 1, 1. * epoch / (end_time - start_time)))

        self.params = params
        # save the best model
        # with open('best_model_params.pkl', 'wb') as f:
        #     pickle.dump(params, f)

        return self


    def fit_with_valid(self, variables, training_sets, validation_sets):
        # check if cost, params, updates are set


        batch_size, learning_rate, index, params = self.batch_size, self.learning_rate, self.index, self.params
        n_train_batches = training_sets[0].get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = validation_sets[0].get_value(borrow=True).shape[0] / batch_size      

        train_model = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                var: data[index*batch_size: (index+1)*batch_size] for var, data in zip(variables, training_sets)
            })

        validate_model = theano.function(
            inputs=[index],
            outputs=self.error,
            givens={
                var: data[index*batch_size: (index+1)*batch_size] for var, data in zip(variables, validation_sets)
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

        return self
        