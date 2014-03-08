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
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.costs.cost import Cost
from pylearn2.datasets.mnist import MNIST
from pylearn2.space import VectorSpace
from pylearn2.costs.cost import SumOfCosts


theano.config.compute_test_value = 'off'

class HonglakLeeSparse(Cost):
    
    def __init__(self, p=0.02):
        self.p = p
    
    def expr(self, model, data):
        
        v = data
        p_h_given_v_matrix = model.propup(v)[-1]
        sum_meta = T.square(self.p - T.mean(p_h_given_v_matrix, axis=0, dtype=theano.config.floatX))
        expr = T.sum(sum_meta)
        
        return expr
    
    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())

class MyContrastiveDivergence(Cost):
    
    def __init__(self, k, chain_num=None): # CD-k
        # k: CD-k
        self.k = k
        self.chain_num = chain_num
	
    def expr(self, model, data):
        return None
        
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()

class MyCD_for(MyContrastiveDivergence):
    
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        #chain_start = pos_v
        v_samples = pos_v
        #print 'v_samples', v_samples.ndim
        for i in xrange(self.k):
            v_samples = model.gibbs_vhv(v_samples)[-1]
        #[act_hids, hid_mfs, hid_samples, act_vis, vis_mfs, vis_samples], updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        #outputs_info=[None, None, None, None, None, chain_start], non_sequences=None, n_steps=self.k)
        neg_v = v_samples
        
        cost = -(- model.free_energy(pos_v).mean() + model.free_energy(neg_v).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, neg_v])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates
    
class MyCD_scan(MyContrastiveDivergence):
    
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        #chain_start = pos_v
        v_samples = pos_v
        #print 'v_samples', v_samples.ndim
        [act_hids, hid_mfs, hid_samples, act_vis, vis_mfs, vis_samples], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, None, None, v_samples], non_sequences=None, n_steps=self.k)
        neg_v = vis_samples[-1]
        
        cost = -(- model.free_energy(pos_v).mean() + model.free_energy(neg_v).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, neg_v])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates

class MyCD_energy_scan(MyContrastiveDivergence):
    
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        pos_h = model.sample_h_given_v(pos_v)[-1]
        #chain_start = pos_v
        h_samples = pos_h
        #print 'v_samples', v_samples.ndim
        [act_vis, vis_mfs, vis_samples, act_hids, hid_mfs, hid_samples], scan_updates = theano.scan(fn = model.gibbs_hvh, sequences=None, 
		                        outputs_info=[None, None, None, None, None, h_samples], non_sequences=None, n_steps=self.k)
        neg_v = vis_samples[-1]
        neg_h = hid_samples[-1]
        
        cost = -(- model.energy(pos_v, pos_h).mean() + model.energy(neg_v, neg_h).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, pos_h, neg_v, neg_h])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates
    
class MyCD_free_energy_scan(MyContrastiveDivergence):
    
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        #pos_h = model.sample_h_given_v(pos_v)[-1]
        #chain_start = pos_v
        #h_samples = pos_h
        #print 'v_samples', v_samples.ndim
        [act_hids, hid_mfs, hid_samples, act_vis, vis_mfs, vis_samples], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, None, None, pos_v], non_sequences=None, n_steps=self.k)
        neg_v = vis_samples[-1]
        #neg_h = hid_samples[-1]
        
        cost = -(- model.free_energy(pos_v).mean() + model.free_energy(neg_v).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, neg_v])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates
    
class MyPCD_for(MyContrastiveDivergence):
    
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        chain_start = theano.shared(numpy.zeros(shape=(self.chain_num, model.n_vis)), name=None, borrow=True)
        v_samples = chain_start
        
        for i in xrange(self.k):
            v_samples = model.gibbs_vhv(v_samples)[-1]
        chain_end = v_samples
        #print 'chain_end', chain_end.ndim
        chain_updates = {}
        chain_updates[chain_start] = chain_end
        
        pos_v = data
        #neg_v = self.get_neg_v(model)
        
        cost = -(- model.free_energy(pos_v).mean() + model.free_energy(chain_end).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[chain_end])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(chain_updates) # manual added

        return gradients, updates
	
class MyPCD_scan(MyContrastiveDivergence):
    
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        chain_start = theano.shared(numpy.zeros(shape=(self.chain_num, model.n_vis), dtype=theano.config.floatX), name='chain_start', borrow=True)
        
        [act_hids, hid_mfs, hid_samples, act_vis, vis_mfs, vis_samples], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, None, None, chain_start], non_sequences=None, n_steps=self.k)
    
        chain_end = vis_samples[-1]
        scan_updates[chain_start] = chain_end
        
        pos_v = data 
        
        cost = -(- model.free_energy(pos_v).mean() + model.free_energy(chain_end).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, chain_end])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # manual added

        return gradients, updates
	
# Is that necessary to inherit Layer class??
class MyRBM(Model, Block):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, n_vis, n_hid, W=None, h_bias=None, v_bias=None, numpy_rng=None,theano_rng=None):
        Model.__init__(self) # self.names_to_del = set(); self._test_batch_size = 2
        Block.__init__(self) # self.fn = None; self.cpu_only = False

        self.n_vis = n_vis
        self.n_hid = n_hid
        
        self.input_space = VectorSpace(dim=self.n_vis) # add input_space
        self.output_space = VectorSpace(dim=self.n_hid) # add output_space

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(seed=19900418)
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

        if h_bias is None:
            # create shared variable for hidden units bias
            h_bias = theano.shared(value=numpy.zeros(n_hid, dtype=theano.config.floatX), name='h_bias', borrow=True)

        if v_bias is None:
            # create shared variable for visible units bias
            v_bias = theano.shared(value=numpy.zeros(n_vis, dtype=theano.config.floatX), name='v_bias', borrow=True)

        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias

        self._params = [self.W, self.h_bias, self.v_bias]
        
    def get_monitoring_data_specs(self):
	    return (self.get_input_space(), self.get_input_source())
		
    def get_monitoring_channels(self, data):
        v = data
        #H = self.sample_h_given_v(v)[1]
        #h = H.mean(axis=0)
        channels = {}

        #channels =  { 'bias_hid_min' : T.min(self.h_bias),
        #         'bias_hid_mean' : T.mean(self.h_bias),
        #         'bias_hid_max' : T.max(self.h_bias),
        #         'bias_vis_min' : T.min(self.v_bias),
        #         'bias_vis_mean' : T.mean(self.v_bias),
        #         'bias_vis_max': T.max(self.v_bias),
        #         'h_min' : T.min(h),
        #         'h_mean': T.mean(h),
        #         'h_max' : T.max(h),
                 #'W_min' : T.min(self.weights),
                 #'W_max' : T.max(self.weights),
                 #'W_norms_min' : T.min(norms),
                 #'W_norms_max' : T.max(norms),
                 #'W_norms_mean' : T.mean(norms),
        #}
		# recon_error
        channel_name = 'recon_error'
        p_v_given_h, v_sample = self.gibbs_vhv(v)[4:]
        recon_error = ((p_v_given_h - v) ** 2).mean(axis=0).sum()
        channels[channel_name] = recon_error
        
        #pos_v = data
        #[h_act, h_mean, h_sample, v_act, v_mean, v_sample], scan_updates = theano.scan(fn = self.gibbs_vhv, sequences=None, 
		#                        outputs_info=[None, None, None, None, None, pos_v], non_sequences=None, n_steps=1)
        #pos_h = h_sample[0]
        #neg_v = v_sample[-1]
        #neg_h = self.sample_h_given_v(v_sample[-1])[-1]
        #cost = -(- self.energy(pos_v, pos_h).mean() + self.energy(neg_v, neg_h).mean())
        #channels['energy_cost'] = cost
        
        #chain_start = theano.shared(numpy.zeros(shape=(20, self.n_vis), dtype=theano.config.floatX), name='chain_start', borrow=True)
        #[h_act, h_mean, h_sample, v_act, v_mean, v_sample], scan_updates = theano.scan(fn = self.gibbs_vhv, sequences=None, 
		#                        outputs_info=[None, None, None, None, None, chain_start], non_sequences=None, n_steps=1)
        #chain_end = v_sample[-1]
        #scan_updates[chain_start] = chain_end
        #pos_v = data 
        #cost = -(- self.free_energy(pos_v).mean() + self.free_energy(chain_end).mean())
        #channels['free_enegy_cost'] = cost
        
		#pseudo_likelihood
        #channel_name = 'pseudo_likelihood'
        #bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        #xi = T.round(v)
        #print 'xi',xi.ndim
        #fe_xi = self.free_energy(xi)
        #xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        #print 'xi_flip', xi_flip.ndim
        #fe_xi_flip = self.free_energy(xi_flip)
        #cost = T.mean(self.n_vis * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        #updates[bit_i_idx] = (bit_i_idx + 1) % self.n_vis
        #channels[channel_name] = cost
        
        return channels
    
    def energy(self, v, h):
        W, c, b = self.get_params()
        #energy = - (T.dot(v, b) + T.dot(h, c) + T.dot(T.dot(v, W), h.T) * T.eye(n=v.shape[0], m=h.shape[0]).sum(axis=0))
        energy = - (T.dot(v, b) + T.dot(h, c) + (T.dot(v, W) * h).sum(axis=1))
        #energy = - (T.dot(v, b) + T.dot(h, c) + T.dot((T.dot(v, W)).T, h))
        return energy

    def free_energy(self, v_sample):
        #print 'free_energy'
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.h_bias
        #print 'wx_b', wx_b.ndim
        v_bias_term = T.dot(v_sample, self.v_bias)
        softplus_term = T.sum(T.nnet.softplus(wx_b), axis=1)
        return - v_bias_term - softplus_term

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.h_bias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.v_bias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
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
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]
				
	# interface for pylearn2.model.mlp PretraindLayer
    def upward_pass(self, state_below):
        return self.propup(state_below)[1]
    
    # default cost is cd-1
    def get_default_cost(self):
        return MyCD_free_energy_scan(k=1)
    
    def make_dataset(self, dataset,sample=False): # use rbm as a feature extractor, daatset pass through the filter and produce a new datset
        
        orin = T.matrix()
        if sample == False:
            f = theano.function([orin], self.propup(orin)[-1])
        else:
            f = theano.function([orin], self.sample_h_given_v(orin)[-1])
        X_new = f(dataset.X)
        new_ds = DatasetFromDesign(design_matrix=X_new, label=dataset.y)
        #print new_ds.__dict__
        #print X_new.shape
        return new_ds

    

