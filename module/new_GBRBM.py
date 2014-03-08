# -*- coding: utf-8 -*-  
import time
import os
from itertools import izip
import copy

import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict

from pylearn2.base import Block
from pylearn2.models.model import Model
#from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.costs.cost import Cost
from pylearn2.models.mlp import MLP, Layer
from pylearn2.datasets.mnist import MNIST
from pylearn2.space import VectorSpace
#from pylearn2.costs.cost import SumOfCosts
from pylearn2.utils import sharedX, as_floatX

from module.dataset_from_design import DatasetFromDesign

#Honglak Lee set sigma to 0.4 and 0.05 for the first and second layers
#we used p = 0.02 and 0.05 for the first and second layers.
# lambda = 1/p in each layer

theano.config.compute_test_value = 'off'
#assert theano.config.floatX == 'float64'

class HonglakLeeSparse(Cost):
    
    def __init__(self, p=0.02, lambda_=None):
        
        self.p = as_floatX(p)
        if lambda_ is None:
            self.lambda_ = as_floatX(1 / self.p)
    
    def expr(self, model, data):
        
        v = data
        p_h_given_v_matrix = model.propup(v)
        sum_meta = (self.p - T.mean(p_h_given_v_matrix, axis=0, dtype=theano.config.floatX))**2
        expr = T.sum(sum_meta, dtype=theano.config.floatX)
        return self.lambda_ * expr
        #return T.cast(self.lambda_ * expr, dtype=theano.config.floatX)
        #return None
    
    def get_data_specs(self, model):
        
        return (model.get_input_space(), model.get_input_source())
    #重写get_gradient()，原因是Honglak Lee指出稀疏惩罚项只用来影响偏置(h_bias)的导数，经实验证明这样做似乎确实可以提高表示的稀疏度
    def get_gradients(self, model, data, ** kwargs):
        """"""
        v = data
        mean_matrix = model.propup(v)
        part_j = self.p - mean_matrix.mean(axis=0)
        part_i1_matrix = mean_matrix * (1. - mean_matrix)
        part_i = T.dot(v.T, part_i1_matrix)
        part_orin = part_i * part_j #矩阵右乘一个行向量
        coeff_w = -2. * model.coeff / v.shape[0]
        gW = coeff_w * part_orin #HL sparse项产生的梯度，不含lambda_
                
        part_j1 = part_j
        part_j2 = part_i1_matrix.mean(axis=0)
        coeff_c = -2. * model.coeff # * part_j1 * part_j2
        gc = coeff_c * part_j1 * part_j2

        W, c, b = list(model.get_params())

        #gradients = OrderedDict(izip([W, c], [1/self.p*gW, 1/self.p*gc]))
        gradients = OrderedDict(izip([c], [T.cast(1/self.p*gc, dtype=theano.config.floatX)]))

        updates = OrderedDict()
        
        #updates.update(scan_updates) # add scan_updates

        return gradients, updates

class MyContrastiveDivergence(Cost):
    
    def __init__(self, k, chain_num=None): # CD-k
        """k: CD-k"""
        self.k = k
        self.chain_num = chain_num
	
    def expr(self, model, data):
        
	    return None
        
    def get_data_specs(self, model):
        
        return model.get_monitoring_data_specs()
    
class MyCD_scan(MyContrastiveDivergence):
    """基于energy的cd-k算法，其中抽样的马氏链中使用抽样sample，在近似计算代价函数关于参数的偏导数时，h使用概率期望。v使用抽样sample
       理论上和基于free energy的cd-k算法是完全相同的"""
    def get_gradients(self, model, data, ** kwargs):
        """cd算法是近似计算导数，而非直接求导，因此重写get_gradient()"""
        pos_v = data
        #v_samples = data
        [h_mean, h_sample, v_mean, v_sample], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, pos_v], non_sequences=None, n_steps=self.k)
        pos_h = h_mean[0]
        neg_v = v_sample[-1]
        neg_h = model.propup(neg_v)
        
        cost = -(- model.energy(pos_v, pos_h).mean() + model.energy(neg_v, neg_h).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, pos_h, neg_v, neg_h])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates
	
# Is that necessary to inherit Layer class??
class MyGaussianBinaryRBM(Model, Block):
    """Gaussian-Binary Restricted Boltzmann Machine (GB-RBM)  """
    def __init__(self, n_vis, n_hid, sigma=0.4, W=None, h_bias=None, v_bias=None, numpy_rng=None,theano_rng=None):
        """ """
        Model.__init__(self) # self.names_to_del = set(); self._test_batch_size = 2
        Block.__init__(self) # self.fn = None; self.cpu_only = False

        self.n_vis = n_vis
        self.n_hid = n_hid
        self.sigma = sigma
        self.coeff = 1. / (self.sigma**2) #coefficient
        
        self.input_space = VectorSpace(dim=self.n_vis) # add input_space
        self.output_space = VectorSpace(dim=self.n_hid) # add output_space

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(seed=19920130)
        self.numpy_rng = numpy_rng

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        if W is None:
            init_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hid + n_vis)),
                      high=4 * numpy.sqrt(6. / (n_hid + n_vis)),
                      size=(n_vis, n_hid)),
                      dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=init_W, name='W', borrow=True)
        else:
            assert isinstance(W, theano.tensor.sharedvar.TensorSharedVariable)
            assert W.get_value().ndim == 2
            
        if h_bias is None:
            # create shared variable for hidden units bias
            h_bias = theano.shared(value=numpy.zeros(n_hid, dtype=theano.config.floatX), name='h_bias', borrow=True)
        else:
            assert isinstance(h_bias, theano.tensor.sharedvar.TensorSharedVariable)
            assert h_bias.get_value().ndim == 1

        if v_bias is None:
            # create shared variable for visible units bias
            v_bias = theano.shared(value=numpy.zeros(n_vis, dtype=theano.config.floatX), name='v_bias', borrow=True)
        else:
            assert isinstance(W, theano.tensor.sharedvar.TensorSharedVariable)
            assert v_bias.get_value().ndim == 1

        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias

        self._params = [self.W, self.h_bias, self.v_bias]
        
    def get_monitoring_data_specs(self):
        
	    return (self.get_input_space(), self.get_input_source())
		
    def get_monitoring_channels(self, data):
        
        channels = OrderedDict()
        v = data
		# recon_error
        channel_name = 'recon_error'
        p_v_given_h, v_sample = self.gibbs_vhv(v)[2:]
        recon_error = ((p_v_given_h - v) ** 2).mean(axis=0).sum()
        channels[channel_name] = recon_error
        
        channel_name = 'hidden_sparseness' #隐层表示稀疏度
        h = self.propup(v)
        sparseness_vector = (T.sqrt(self.n_hid) - T.sum(T.abs_(h), axis=1) / T.sqrt(T.sum(h**2, axis=1))) / (T.sqrt(self.n_hid) - 1.)
        assert sparseness_vector.ndim == 1
        channels[channel_name] = T.mean(sparseness_vector)
        
        channel_name = 'W_sparseness' #权值矩阵稀疏度
        W = self.W.T
        sparseness_vector = (T.sqrt(self.n_vis) - T.sum(T.abs_(W), axis=1) / T.sqrt(T.sum(W**2, axis=1))) / (T.sqrt(self.n_vis) - 1.)
        assert sparseness_vector.ndim == 1
        channels[channel_name] = T.mean(sparseness_vector)
        
        return channels
    
    def energy(self, v, h):
        """Function to compute energy"""
        energy = 0.5 * self.coeff * T.sum(v**2, axis=1) - self.coeff * (T.dot(v, self.v_bias) + T.dot(h, self.h_bias) + (T.dot(v, self.W) * h).sum(axis=1))
        
        return energy

    def free_energy(self, v):
        ''' Function to compute the free energy '''
        v_term = self.coeff * (0.5 * T.sum(v**2, axis=1) - T.dot(v, self.v_bias))
        softplus_act = self.coeff * (T.dot(v, self.W) + self.h_bias)
        softplus_term = - T.sum(T.nnet.softplus(softplus_act), axis=1)
        free_energy = v_term + softplus_term
        return free_energy

    def propup(self, v):
        """"""
        pre_sigmoid_activation = self.coeff * (T.dot(v, self.W) + self.h_bias)
        return T.nnet.sigmoid(pre_sigmoid_activation)

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        h0_mean = self.propup(v0_sample)
        h0_sample = self.theano_rng.binomial(n=1, p=h0_mean,
                                             dtype=theano.config.floatX)
        return [h0_mean, h0_sample]

    def propdown(self, h):
        
        Gaussian_mean = self.v_bias + T.dot(h, self.W.T)
        return Gaussian_mean

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.normal(avg=v1_mean, std=self.sigma, dtype=theano.config.floatX)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        h0_mean, h0_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        return [h0_mean, h0_sample, v1_mean, v1_sample]
				
	# interface for pylearn2.model.mlp PretraindLayer
    def upward_pass(self, state_below):
        return self.propup(state_below)
    

        

    
if __name__ == '__main__':
    
    from pylearn2.datasets.mnist import MNIST
    from pylearn2.training_algorithms.sgd import SGD
    from pylearn2.train import Train
    from pylearn2.termination_criteria import MonitorBased
    from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
    from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
    from pylearn2.training_algorithms.sgd import MomentumAdjustor
    from pylearn2.termination_criteria import EpochCounter
    from pylearn2.costs.cost import SumOfCosts
    
    import cPickle
	
    f = open('/home/zanghu/useful_data/dsc_cifar10_preprocessed_patches.pkl')
    dsc_train = cPickle.load(f)
    f.close()
    #monitoring_dataset = {'train': dsm_train, 'valid': dsm_valid, 'test': dsm_test}
    monitoring_dataset = {'train': dsc_train}
	
    rbm_model = MyGaussianBinaryRBM(n_vis=192, n_hid=400)
    
    #cd_cost = MyCD_scan(k=15)
    #total_cost = MyPCD_scan(k=15, chain_num=20)
    total_cost = SumOfCosts(costs=[MyCD_scan(k=1), HonglakLeeSparse(p=0.02)])
    
    alg = SGD(learning_rate=0.001, cost=total_cost, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset,
              #termination_criterion=MonitorBased(channel_name='valid_recon_error', N=10))
              termination_criterion=EpochCounter(max_epochs=15))  
    
    #MonitorBasedLRAdjuster(dataset_name='valid'),MomentumAdjustor(start=1, saturate=20, final_momentum=.99)
    train = Train(dataset=dsc_train, model=rbm_model, algorithm=alg,
            #extensions=[MonitorBasedSaveBest(channel_name='test_recon_error', save_path='my_rbm_1021.pkl')],
            save_path='my_rbm_trainsave_1021.pkl',
            save_freq=10)
    t0 = time.clock()
    train.main_loop()
    t1 = time.clock()
    print 'time elapsed on training is', t1 - t0

