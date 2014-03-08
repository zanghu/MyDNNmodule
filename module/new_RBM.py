#coding: utf8
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
from pylearn2.costs.cost import Cost, SumOfCosts
from pylearn2.models.mlp import MLP, Layer
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX, as_floatX

from module.dataset_from_design import DatasetFromDesign


theano.config.compute_test_value = 'off'

class HonglakLeeSparse(Cost):
    """Honglak Lee nips07"""
    def __init__(self, p=0.02, lambda_=None):
        
        self.p = p
        if lambda_ is None:
            lambda_ = as_floatX(1. / self.p)
        self.lambda_ = lambda_
    
    def expr(self, model, data):
        
        v = data
        p_h_given_v_matrix = model.propup(v)[-1]
        sum_meta = T.square(self.p - T.mean(p_h_given_v_matrix, axis=0, dtype=theano.config.floatX))
        expr = T.sum(sum_meta)
        
        return expr
    
    def get_data_specs(self, model):
        
        return (model.get_input_space(), model.get_input_source())
    
    def get_gradients(self, model, data, ** kwargs):
        
        v = data
        mean_matrix = model.propup(v)
        #======================================================
        part_j = self.p - mean_matrix.mean(axis=0)
        part_i1_matrix = mean_matrix * (1. - mean_matrix)
        #part_i = T.dot(v.T, part_i1_matrix)
        #part_orin = part_i * part_j #矩阵右乘一个行向量
        #coeff_w = -2. *  v.shape[0]
        #gW = coeff_w * part_orin #HL sparse项产生的梯度，不含lambda_
        #=======================================================
        
        part_j1 = part_j
        part_j2 = part_i1_matrix.mean(axis=0)
        gc = -2. * part_j1 * part_j2

        W, c, b = list(model.get_params())

        #gradients = OrderedDict(izip([W, c], [1/self.p*gW, 1/self.p*gc]))
        gradients = OrderedDict(izip([c], [as_floatX(1/self.p*gc)]))

        updates = OrderedDict()

        return gradients, updates
         
    

class MyContrastiveDivergence(Cost):
    """"CD算法的基类"""
    def __init__(self, k, chain_num=None): # CD-k
        # k: CD-k
        self.k = k
        self.chain_num = chain_num
	
    def expr(self, model, data):
        #pos_v = data
        #v_samples = pos_v
        #for i in xrange(self.k):
        #    v_samples = model.gibbs_vhv(v_samples)[-1]
        #neg_v = v_samples
        #neg_log_likelihood = -(- model.free_energy(pos_v).mean() + model.free_energy(neg_v).mean())
        #return neg_log_likelihood
	    return None
        
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()

class MyCD_energy(MyContrastiveDivergence):
    """pos_h, neg_h全部使用抽样值"""
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        pos_h = model.sample_h_given_v(pos_v)[0]
        #print 'v_samples', v_samples.ndim
        [v_mean, v_sample, h_mean, h_sample], scan_updates = theano.scan(fn = model.gibbs_hvh, sequences=None, 
		                        outputs_info=[None, None, None, pos_h], non_sequences=None, n_steps=self.k)
        neg_v = v_sample[-1]
        neg_h = h_sample[-1]
        
        cost = -(- model.energy(pos_v, pos_h).mean() + model.energy(neg_v, neg_h).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, pos_h, neg_v, neg_h])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates
    
class MyCD_free_energy(MyContrastiveDivergence):
    """pos_h，neg_h全部使用mean值"""
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        [h_mean, h_sample, v_mean, v_sample], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, pos_v], non_sequences=None, n_steps=self.k)
        neg_v = v_sample[-1]
        
        cost = -(- model.free_energy(pos_v).mean() + model.free_energy(neg_v).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, neg_v])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates


	
# Is that necessary to inherit Layer class??
class MyRBM(Model, Block):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, n_vis, n_hid, W=None, h_bias=None, v_bias=None, numpy_rng=None,theano_rng=None):
        """"""
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
        """"""
	return (self.get_input_space(), self.get_input_source())
		
    def get_monitoring_channels(self, data):
        """"""
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
        p_v_given_h, v_sample = self.gibbs_vhv(v)[2:]
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
        """计算能量函数"""
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

    def propup(self, v):
        """计算正向传播期望"""
        sigmoid_activation = T.dot(v, self.W) + self.h_bias
        return T.nnet.sigmoid(sigmoid_activation)

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        h0_mean = self.propup(v0_sample)
        h0_sample = self.theano_rng.binomial(n=1, p=h0_mean, dtype=theano.config.floatX)
        return [h0_mean, h0_sample]

    def propdown(self, h):
        """计算反向传播期望"""
        sigmoid_activation = T.dot(h, self.W.T) + self.v_bias
        return T.nnet.sigmoid(sigmoid_activation)

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
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
        """用来兼容mlp"""
        return self.propup(state_below)
    
    def visual_config(self, step=1, dpi=240, start=0, total_num=100, v_shape=(28, 28), v_channels=1, random=False):
        """"""
        assert total_num < self.n_hid - start
        self.__dict__.update(locals())
        del self.self
    
    #need to be cooperate with module.my_train_extensions.Visualizer(TrainExtension) instance
    # which shoud be added to extensions prameter of pylearn2..train.Train
    def get_visual(self):
        #step: draw pictures every step epochs
        #total_num: number of hidden units which need to be draw
        #v_shape: shape of input image
        #v_channels: number of channels that each input has
        # random: if True, randomly select total_num hiddenunits, else use the first total_num hidden units
        assert total_num < self.n_hid
        save_path = os.getcwd() + '/visual_pic'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        if hasattr(self, 'epoch_cnt'):
            self.epoch_cnt += 1
        else:
            self.epoch_cnt = 1
        rng = self.numpy_rng
        W = self.W.get_value()
        v = W / ((W**2).sum(axis=1)).reshape(W.shape[0], 1)
        
        if (self.epoch_cnt - 1) % self.step != 0:
            return None
        
        if self.random is True:
            rl = rng.choice(xrange(self.n_hid), self.total_num)
        else:
            rl = range(self.total_num)
        
        l = int(numpy.ceil(numpy.sqrt(self.total_num)))

        for i in xrange(l):
            for j in xrange(l):
                if self.v_channels == 1: #draw gray level img
                    pylab.subplot(l, l, i*l+j+1); pylab.axis('off')
                    pylab.imshow(v[:, rl[i*l+j]].reshape(self.v_shape), cmap=pylab.cm.gray)
                else: # draw RGB img
                    pylab.subplot(l, l, i*l+j+1); pylab.axis('off')
                    pylab.imshow(v[:, rl[i*l+j]].reshape(3, self.v_shape[0], self.v_shape[1]).transpose(1,2,0))

        pylab.savefig(save_path + '/epoch_' + str(self.epoch_cnt) + '.png', dpi=self.dpi) # save img
        
        return None
        
    # default cost is cd-1
    def get_default_cost(self):
        """"""
        #return SumOfCosts(costs=[MyCD_free_energy(k=1), HonglakLeeSparse()])
        #return MyCD_free_energy(k=1)
        return MyCD_energy(k=1)
    
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
    
    def get_sparseness_theano(W):
        """计算矩阵W每一列的稀疏程度的平均值，返回代表稀疏度的符号变量"""
        n = W.shape[0]
        W = self.W.T
        sparseness_vector = (T.sqrt(n) - T.sum(T.abs_(W), axis=1) / T.sqrt(T.sum(W**2, axis=1))) / (T.sqrt(n) - 1.)
        assert sparseness_vector.ndim == 1
        return T.mean(sparseness_vector)
    
    def get_sparseness_numpy(h):
        """计算矩阵W每一列的稀疏程度的平均值，返回代表稀疏度的数值"""
        n = W.shape[0]
        W = self.W.T
        sparseness_vector = (numpy.sqrt(n) - numpy.sum(numpy.abs(W), axis=1) / numpy.sqrt(numpy.sum(W**2, axis=1))) / (numpy.sqrt(n) - 1.)
        assert sparseness_vector.ndim == 1
        return numpy.mean(sparseness_vector)
