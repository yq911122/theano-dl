import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Layer(object):
    """Base Class for network layer.
    
    Parameters
    ----------
    n_in : int
        dimension of the input to the layer

    n_out : int
        dimension of the output of the layer

    rng : np.random.RandomState, optional (default=None)
        random state for parameter random initation

    W : Theano shared variable, optional (default=None)
        wieght parameters of the layer

    b : Theano shared variable, optional (default=None)
        bias parameters of the layer

    activation : function, optional (default=None)
        activation function

    Attributes
    ----------
    rng : np.random.RandomState

    W : Theano shared variable; wieght parameters of the layer

    b : Theano shared variable; bias parameters of the layer

    params :  List; list that stores layer parameters

    activation : function; activation function
    
    output : Theano variable; layer output
    """
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
        """calculate output of the layer given X as input
        
        Parameters
        ----------
        X : Theano variable; input to the layer

        Return
        ------
        output : Theano variable; output of the layer

        """
        output_val = T.dot(X, self.W) + self.b
        self.output = self.activation(output_val) if self.activation else output_val
        return self.output

class LogitLayer(Layer):
    """layer applying logit activation. Normally, this layer will be as the output layer.

    Parameters
    ----------
    n_in : int
        dimension of the input to the layer

    n_out : int
        dimension of the output of the layer

    rng : np.random.RandomState, optional (default=None)
        random state for parameter random initation

    W : Theano shared variable, optional (default=None)
        wieght parameters of the layer

    b : Theano shared variable, optional (default=None)
        bias parameters of the layer

    Attributes
    ----------
    rng : np.random.RandomState

    W : Theano shared variable; wieght parameters of the layer

    b : Theano shared variable; bias parameters of the layer

    params :  List; list that stores layer parameters

    activation : T.nnet.softmax
    
    output : Theano variable; layer output, also the class probability

    y_pred : Theano variable; predicted class based on the output
    """
    def __init__(self, n_in, n_out, rng=None, W=None, b=None):
        super(LogitLayer, self).__init__(
            n_in=n_in, 
            n_out=n_out, 
            rng=rng,
            W=W, 
            b=b,
            activation=T.nnet.softmax)

    def cal_output(self, X):
        """calculate output of the layer given X as input
        
        Parameters
        ----------
        X : Theano variable; input to the layer

        Return
        ------
        output : Theano variable; output of the layer

        """
        super(LogitLayer, self).cal_output(X)
        
        self.y_pred = T.argmax(self.output, axis=1)
        return self.output

    def get_predict(self):
        """return y_pred

        Return
        ------
        self.y_pred : Theano variable; predicted class based on the output

        """
        return self.y_pred

    def get_predict_proba(self):
        """return predict probability of class

        Return
        ------
        self.output : Theano variable; predict probability of class
        
        """
        return self.output

    def cost(self, y):
        """calculate average negative loglikelyhood as cost

        Parameters
        ----------
        y : Theano variable; test target

        Return
        ------
        NLL : Theano variable; cost
        """
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def error(self, y):
        """calculate mis-classification error

        Parameters
        ----------
        y : Theano variable; type=int, test target

        Return
        ------
        error : Theano variable; mis-classification error
        """
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

class ConvPoolLayer2D(Layer):
    """Convolution and pooling layer for 2d input, like image. In the layer, a pooling operation is followed by a concolution operation.

    Parameters
    ----------
    filter_shape : tuple (n_output_feature_maps, n_input_feature_maps, filter_height, filter_width)

    image_shape : tuple (batch_size, n_input_feature_maps, input_height, input_width)

    poolsize : tuple (height, height)
        size to applying pooling

    rng : np.random.RandomState, optional (default=None)
        random state for parameter random initation

    W : Theano shared variable, optional (default=None)
        wieght parameters of the layer

    b : Theano shared variable, optional (default=None)
        bias parameters of the layer

    activation : function, optional (default=T.tanh)
        activation function

    Attributes
    ----------
    filter_shape : tuple (n_output_feature_maps, n_input_feature_maps, filter_height, filter_width)
    
    image_shape : tuple (batch_size, n_input_feature_maps, input_height, input_width)

    poolsize : tuple (height, height)
        size to applying pooling
    
    rng : np.random.RandomState

    W : Theano shared variable; wieght parameters of the layer

    b : Theano shared variable; bias parameters of the layer

    params :  List; list that stores layer parameters

    activation : activation function
    
    output : Theano variable; layer output, also the class probability
    """
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), rng=None, W=None, b=None, activation=T.tanh):
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

        super(ConvPoolLayer2D, self).__init__(
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
        """calculate output after convolution and pooling operations of the layer given X as input
        
        Parameters
        ----------
        X : Theano variable; input to the layer

        Return
        ------
        output : Theano variable; output of the layer

        """
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

        output_val = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = self.activation(output_val) if self.activation else output_val

        return self.output
        
class dA(Layer):
    """denoising Autoencoder layer. It is actually a 3-layer structure: input-hidden-output, where the dimensions of the input layer and the output layer are the same.
    
    Parameters
    ----------
    n_in : int
        dimension of the input to the layer

    n_hidden : int
        dimension of the n_hidden layer

    rng : np.random.RandomState, optional (default=None)
        random state for parameter random initation

    theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams, optional (default=None)
        random state for corruption calculation    
    
    W : Theano shared variable, optional (default=None)
        wieght parameters of the layer

    bvis : Theano shared variable, optional (default=None)
        bias parameters of the output layer

    bhid : Theano shared variable, optional (default=None)
        bias parameters of the hidden layer

    activation : function, optional (default=T.nnet.sigmoid)
        activation function

    corruption_level : float, optional (default=0.1)
        corruption level of input

    Attributes
    ----------
    rng : np.random.RandomState

    theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams, optional (default=None)
        random state for corruption calculation

    W : Theano shared variable; wieght parameters of the input layer

    b : Theano shared variable; bias parameters of the hidden layer

    W_prime : Theano shared variable; wieght parameters of the hidden layer

    b_prime : Theano shared variable; wieght parameters of the output layer

    params :  List; list that stores layer parameters

    activation : function; activation function
    
    output : Theano variable; layer output

    corruption_level : float, optional (default=0.1)
        corruption level of input

    """
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
        """calculate output of the hidden layer given X as input
        
        Parameters
        ----------
        X : Theano variable; input to the input layer

        Return
        ------
        output : Theano variable; output of the hidden layer

        """
    	return super(dA, self).cal_output(X)

    def get_vis_output(self, X):
        """calculate output of the output layer given X as input of the hidden layer
        
        Parameters
        ----------
        X : Theano variable; input to the hidden layer

        Return
        ------
        output : Theano variable; output of the output layer

        """
    	output_val = T.dot(X, self.W_prime) + self.b_prime
        self.output = self.activation(output_val) if self.activation else output_val
        return self.output

    def cal_output(self, X):
        """same with get_hidden_output(self, X)
        """
        return self.get_hidden_output(X)

    def cal_final_output(self, X):
        """calculate output of the output layer given X as input of the input layer
        
        Parameters
        ----------
        X : Theano variable; input to the input layer

        Return
        ------
        output : Theano variable; output of the output layer

        """
    	y = self.get_hidden_output(X)
        return self.get_vis_output(y)

    def cost(self, x):
        """calculate average entropy as cost

        Parameters
        ----------
        x : Theano variable; input

        Return
        ------
        cost : Theano variable; cost
        """
        tilde_x = self.get_corrupted_input(x)
        y = self.cal_final_output(tilde_x)
        L = - T.sum(x * T.log(y) + (1 - x) * T.log(1 - y), axis=1)
    	return T.mean(L)

    def get_corrupted_input(self, input):
        """get corrupted input

        Parameters
        ----------
        input : Theano variable; input

        Return
        ------
        corrupted_input : Theano variable; corrupted_input
        """

        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - self.corruption_level,
                                        dtype=theano.config.floatX) * input

class RBMLayer(Layer):
    """restricted boltzmann machine
    
    Parameters
    ----------
    n_visible : int
        dimension of the input to (also output of) the layer

    n_hidden : int
        dimension of the n_hidden layer

    rng : np.random.RandomState, optional (default=None)
        random state for parameter random initation

    theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams, optional (default=None)
        random state for corruption calculation    
    
    W : Theano shared variable, optional (default=None)
        wieght parameters of the layer

    bvis : Theano shared variable, optional (default=None)
        bias parameters of the output (also input) layer

    bhid : Theano shared variable, optional (default=None)
        bias parameters of the hidden layer

    activation : function, optional (default=T.nnet.sigmoid)
        activation function

    k : int, steps for CD-K sampling

    Attributes
    ----------
    rng : np.random.RandomState

    theano_rng : theano.sandbox.rng_mrg.MRG_RandomStreams, optional (default=None)
        random state for corruption calculation

    W : Theano shared variable; wieght parameters of the input layer

    b : Theano shared variable; bias parameters of the hidden layer

    b_prime : Theano shared variable; wieght parameters of the output layer

    params :  List; list that stores layer parameters

    activation : function; activation function
    
    output : Theano variable; layer output

    corruption_level : float, optional (default=0.1)
        corruption level of input

    """
    def __init__(
        self,
        n_visible,
        n_hidden,
        W=None,
        bhid=None,
        bvis=None,
        rng=None,
        theano_rng=None,
        activation=T.nnet.sigmoid,
        k=1
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if rng is None:
            rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2 ** 30))

        super(RBMLayer, self).__init__(
        	n_in=n_visible,
        	n_out=n_hidden,
        	rng=rng, 
        	W=W, 
        	b=bhid, 
        	activation=activation
        	)       

        if bvis is None:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='bvis',
                borrow=True
            )

        self.bhid = self.b
        self.bvis = bvis
        self.theano_rng = theano_rng

        self.params = [self.W, self.bhid, self.bvis]
        self.k = k

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.bhid
        bvis_term = T.dot(v_sample, self.bvis)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - bvis_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.bhid
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
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.bvis
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

    def get_cost_updates(self, X, lr=0.1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM


        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but

        """

        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(X)

        chain_start = ph_sample

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

        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(X)) - T.mean(
            self.free_energy(chain_end))

        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * lr

        monitoring_cost = self.get_reconstruction_cost(X, pre_sigmoid_nvs[-1])
        
        return monitoring_cost, updates

    def get_reconstruction_cost(self, X, pre_sigmoid_nv):
        """Approximation to the reconstruction error
        """

        cross_entropy = T.mean(
            T.sum(
                X * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - X) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy