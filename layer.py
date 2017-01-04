import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
    def __init__(self, n_in, n_hidden, rng=None, theano_rng=None, W=None, bvis=None, bhid=None, activation=T.nnet.sigmoid, corruption_level=0.1):
        super(dA, self).__init__(
            n_in=n_in, 
            n_out=n_hidden, 
            rng=rng,
            W=W, 
            b=bhid,
            activation=activation)
        
        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_in,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        self.b_prime = bvis
        self.W_prime = self.W.T

        self.params += [self.b_prime]

        if not theano_rng:
            theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.theano_rng = theano_rng
        self.corruption_level = corruption_level

    def get_hidden_output(self, X):
    	return super(dA, self).cal_output(X)

    def get_vis_output(self, X):
    	output_val = T.dot(X, self.W_prime) + self.b_prime
        self.output = self.activation(output_val) if self.activation else output_val
        return self.output

    def cal_output(self, X):
        return self.get_hidden_output(X)

    def cal_final_output(self, X):
    	y = self.get_hidden_output(X)
        return self.get_vis_output(y)

    def cost(self, x):
        tilde_x = self.get_corrupted_input(x, self.corruption_level)
        y = self.cal_final_output(tilde_x)
        L = - T.sum(x * T.log(y) + (1 - x) * T.log(1 - y), axis=1)
    	return T.mean(L)

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

class RBMLayer(Layer):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        n_visible,
        n_hidden,
        W=None,
        hbias=None,
        vbias=None,
        rng=None,
        theano_rng=None,
        activation=T.nnet.sigmoid,
        k=1
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if rng is None:
            # create a number generator
            rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2 ** 30))

        super(RBMLayer, self).__init__(
        	n_in=n_visible,
        	n_out=n_hidden,
        	rng=rng, 
        	W=W, 
        	b=hbias, 
        	activation=activation
        	)       

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )


        # initialize input layer for standalone RBM or layer0 of DBN

        self.hbias = self.b
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        self.k = k

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
    def get_cost_updates(self, X, lr=0.1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM


        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(X)

        chain_start = ph_sample

        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=self.k,
            name="gibbs_hvh"
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(X)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )

        # reconstruction cross-entropy is a better proxy for CD
        monitoring_cost = self.get_reconstruction_cost(X, pre_sigmoid_nvs[-1])


        return monitoring_cost, updates
        # end-snippet-4

    def get_reconstruction_cost(self, X, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(
                X * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - X) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy