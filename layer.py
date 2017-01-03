import theano
import theano.tensor as T
import numpy as np

class Layer(object):
    """docstring for Layer"""
    def __init__(self, n_in, n_out, rng=None, W=None, b=None, activation=None):
        super(Layer, self).__init__()
        if rng is None:
            self.rng = np.random.RandomState(1123)

        if W is None:
            W_values = np.asarray(
                self.rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.params = [self.W, self.b]

        self.activation = activation


    def cal_output(self, X):        
        output_val = T.dot(X, self.W) + self.b
        self.output = self.activation(output_val) if self.activation else output_val
        return self.output


class LogitLayer(Layer):
    """docstring for LogitLayer"""
    def __init__(self, n_in, n_out, rng=None, W=None, b=None):
        super(LogitLayer, self).__init__(
            n_in=n_in, 
            n_out=n_out, 
            rng=rng,
            W=W, 
            b=b,
            activation=T.nnet.softmax)

    def cal_output(self, X):
        super(LogitLayer, self).cal_output(X)
        # self.input = X
        
        # output_val = T.dot(self.input, self.W) + self.b
        # self.output = self.activation(output_val) if self.activation else output_val
        self.y_pred = T.argmax(self.output, axis=1)
        return self.output

    def get_predict(self):
        return self.y_pred

    def get_predict_proba(self):
        return self.output

    def cost(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def error(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

class ConvPoolLayer(Layer):
    """docstring for ConvPoolLayer"""
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), rng=None, W=None, b=None, activation=None):
        if rng is None:
            self.rng = np.random.RandomState(1123)
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))

        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                np.asarray(
                    self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if b is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        super(ConvPoolLayer, self).__init__(
            n_in=fan_in, 
            n_out=fan_out, 
            rng=self.rng,
            W=W, 
            b=b,
            activation=activation)

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize

    def cal_output(self, X):
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=X,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        output_val = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = self.activation(output_val) if self.activation else output_val

        return self.output
        
class dA(Layer):
    """docstring for dA"""
    def __init__(self, n_in, n_hidden, rng=None, theano_rng=None, W=None, bvis=None, bhid=None, activation=T.nnet.sigmoid):
        super(dA, self).__init__(
            n_in=n_in, 
            n_out=n_hidden, 
            rng=rng,
            W=W, 
            b=bhid,
            activation=activation)
        
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        self.b_prime = bvis
        self.W_prime = self.W.T

        self.params += [self.b_prime]

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng


    def cal_output(self, X):
        super(dA, self).cal_output(X)     
        output_val = T.dot(self.output, self.W_prime) + self.b_prime
        self.output = self.activation(output_val) if self.activation else output_val
        return self.output

    def cost(self, x, corruption_level):
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.cal_output(tilde_x)
        return - T.sum(x * T.log(y) + (1 - x) * T.log(1 - y), axis=1)
    
    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input