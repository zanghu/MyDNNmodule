#coding=utf8
import time
from itertools import izip

import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict

from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.costs.cost import Cost
from pylearn2.models.mlp import MLP, Layer
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX, as_floatX


theano.config.compute_test_value = 'off'

class ContrastiveDivergence(Cost): #注意rsm模型没有pcd算法
    
    def __init__(self, k=1): # CD-k
        self.k = k
	
    def expr(self, model, data):
	    return None
    
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()


class MyCD_energy_scan(ContrastiveDivergence):
    """完全基于能量函数的，标准的cd-k参数更新，遗憾的事不明原因的效果不好，至少从重构误差值上看是这样"""
    def get_gradients(self, model, data, ** kwargs):
        """cdk算法近似计算不能直接用代价函数关于参数求导，重写"""
        pos_v = data
        [h_mean, h_sample, v_mean, v_sample], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, pos_v], non_sequences=None, n_steps=self.k)
        
        pos_h = h_mean[0]
        neg_v = v_sample[-1]   
        neg_h = model.propup(v_sample[-1])
        
        cost = -(- model.energy(pos_v, pos_h).mean() + model.energy(neg_v, neg_h).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, pos_h, neg_v, neg_h])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates
    
    def get_monitoring_channels(self, model, data, **kwargs):
         
        channels = OrderedDict()
        return channels

class CDk(ContrastiveDivergence):
    """重写后，保留原有错误的cd-k参数更新"""
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        #chain_start = theano.shared(numpy.zeros(shape=(self.chain_num, model.n_vis), dtype=theano.config.floatX), name='chain_start', borrow=True)
        #v_sample = pos_v
        [h_mean, h_sample, v_mean, v_sample], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, pos_v], non_sequences=None, n_steps=self.k)
        pos_h = h_mean[0]
        neg_v = v_sample[-1]
        neg_h = model.propup(v_sample[-1])
        
        #cost = -(- model.energy(pos_v, pos_h).mean() + model.energy(neg_v, neg_h).mean())

        #params = list(model.get_params())

        #grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_h, neg_v, neg_h]

        #gradients = OrderedDict(izip(params, grads))
        gradients = OrderedDict()
        D = data.sum(axis=1)
        gradients[model.W] = T.cast(-T.dot(pos_v.T, pos_h) / data.shape[0] + T.dot(neg_v.T, neg_h) / data.shape[0], dtype=theano.config.floatX)
        gradients[model.h_bias] = T.cast(- pos_h.mean(axis=0) + neg_h.mean(axis=0), dtype=theano.config.floatX) #效果好但错误的参数更新
        #gradients[model.h_bias] = T.cast(- (D.dimshuffle(0, 'x') * pos_h).mean(axis=0) + (D.dimshuffle(0, 'x') * neg_h).mean(axis=0), dtype=theano.config.floatX) #效果差但正确的参数更新
        gradients[model.v_bias] = T.cast(- pos_v.mean(axis=0) + neg_v.mean(axis=0), dtype=theano.config.floatX)
        

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates
    
    def get_monitoring_channels(self, model, data, **kwargs):
         
        channels = OrderedDict()
        return channels
	
# Is that necessary to inherit Layer class??
class ReplicatedSoftmaxRBM(Model, Block):
    """Replicated Softmax RBM (RSM)  """
    def __init__(self, n_vis, n_hid, W=None, h_bias=None, v_bias=None, numpy_rng=None,theano_rng=None):
        Model.__init__(self) # self.names_to_del = set(); self._test_batch_size = 2
        Block.__init__(self) # self.fn = None; self.cpu_only = False

        self.n_vis = n_vis
        self.n_hid = n_hid
        
        self.input_space = VectorSpace(dim=self.n_vis) # add input_space
        self.output_space = VectorSpace(dim=self.n_hid) # add output_space

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(seed=20880130)
        self.numpy_rng = numpy_rng

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        if W is None:
            #init_W = numpy.asarray(numpy_rng.uniform(
            #          low= - numpy.sqrt(6. / (n_hid + n_vis)),
            #          high= numpy.sqrt(6. / (n_hid + n_vis)),
            #          size=(n_vis, n_hid)),
            #          dtype=theano.config.floatX)
            init_W = 0.001 * numpy.random.randn(n_vis, n_hid)
            W = theano.shared(value=init_W, name='W', borrow=True)

        if h_bias is None:
            #h_bias = theano.shared(value=numpy.zeros(n_hid, dtype=theano.config.floatX), name='h_bias', borrow=True)
            h_bias = theano.shared(value=0.001 * numpy.random.randn(n_hid), name='h_bias', borrow=True)

        if v_bias is None:
            #v_bias = theano.shared(value=numpy.zeros(n_vis, dtype=theano.config.floatX), name='v_bias', borrow=True)
            v_bias = theano.shared(value=0.001 * numpy.random.randn(n_vis), name='v_bias', borrow=True)

        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias

        self._params = [self.W, self.h_bias, self.v_bias]
        
    def get_monitoring_data_specs(self):
	    return (self.get_input_space(), self.get_input_source())
		
    def get_monitoring_channels(self, data):
        v = data
        channels = OrderedDict()
        
		# recon_error
        v_sample = self.gibbs_vhv(v)[-2]
        #v_sample = self.gibbs_vhv(v)[-1]
        #recon_error = ((v_sample - v) ** 2).sum(axis=1).mean() / v.shape[1]
        recon_error = ((v_sample - v) ** 2).mean()
        channels['recon_error'] = T.cast(recon_error, dtype=theano.config.floatX)
        
        #check
        delta = v.sum(axis=1) - self.gibbs_vhv(v)[-1].sum(axis=1)
        num = T.neq(delta, T.zeros(shape=delta.shape, dtype=delta.dtype)).sum()
        channels['error_num'] = T.cast(num, dtype=theano.config.floatX)
        return channels
        #return None
    
    def energy(self, v, h): # rsm specified
        """symbol representation of energy function"""
        W, c, b = self.get_params()
        D = T.sum(v, axis=1) # .dimshuffle(0, 'x')
        #energy = - (T.dot(v, b) + D * (T.dot(h, c)) + T.dot(T.dot(v, W), h.T) * T.eye(n=v.shape[0], m=h.shape[0]).sum(axis=0))
        energy = - (T.dot(v, b) + D * (T.dot(h, c)) + (T.dot(v, W) * h).sum(axis=1))
        return energy

    #def free_energy(self, v): # rsm specified
    #    D = T.sum(v, axis=1)
    #    wx_b = T.dot(v, self.W) + self.h_bias * D.dimshuffle(0, 'x')
    #    #wx_b = T.dot(v, self.W) + self.h_bias
    #    v_bias_term = T.dot(v, self.v_bias)
    #    softplus_term = T.sum(T.nnet.softplus(wx_b), axis=1)
    #    return - v_bias_term - softplus_term
    def propup(self, v1):
        """用于只需要计算可视层v的均值，不需要抽样的情况"""
        D = T.sum(v1, axis=1)
        act_h1 = T.dot(v1, self.W) + D.dimshuffle(0, 'x') * self.h_bias # sigmoid activision
        h1_mean = T.nnet.sigmoid(act_h1)
        return h1_mean
    
    def propdown(self, h1, v1):
        """用于只需要计算隐藏层h的均值，不需要抽样的情况"""
        numerator = T.exp(self.v_bias + T.dot(h1, self.W.T))
        denominator = numerator.sum(axis=1)
        v2_pdf = numerator / (denominator.dimshuffle(0, 'x'))

        D = v1.sum(axis=1)
        v2_mean = D.dimshuffle(0, 'x') * v2_pd
        return v2_mean
    
    def sample_h_given_v(self, v1): # rsm specified
        """"""
        D = T.sum(v1, axis=1)
        act_h1 = T.dot(v1, self.W) + D.dimshuffle(0, 'x') * self.h_bias # sigmoid activision
        h1_mean = T.nnet.sigmoid(act_h1)
        h1_sample = self.theano_rng.binomial(size=None, n=1, p=h1_mean, dtype=theano.config.floatX)
        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h1, v1): # rsm specified
        """"""
        numerator = T.exp(self.v_bias + T.dot(h1, self.W.T))
        denominator = numerator.sum(axis=1)
        v2_pdf = numerator / (denominator.dimshuffle(0, 'x'))

        D = v1.sum(axis=1)
        v2_mean = D.dimshuffle(0, 'x') * v2_pdf
        v2_sample = self.theano_rng.multinomial(size=None, n=D, pvals=v2_pdf, dtype=theano.config.floatX)
       
        return [v2_mean, v2_sample]

    def gibbs_vhv(self, v1): # rsm specified
        """"""
        h1_mean, h1_sample = self.sample_h_given_v(v1)
        v2_mean, v2_sample = self.sample_v_given_h(h1_sample, v1)
        return [h1_mean, h1_sample, v2_mean, v2_sample]
				
	# interface for pylearn2.model.mlp PretraindLayer
    def upward_pass(self, state_below, sample=False):
        
        if sample is False:
            return self.sample_h_given_v(state_below)[0]
        else:
            return self.sample_h_given_v(state_below)[1]

if __name__ == '__main__':
    
    from pylearn2.training_algorithms.sgd import SGD
    from pylearn2.train import Train
    from pylearn2.termination_criteria import MonitorBased
    from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
    from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
    from pylearn2.training_algorithms.sgd import MomentumAdjustor
    from pylearn2.termination_criteria import EpochCounter
    import time
    
    from news20 import News20
	
    ds20_train = News20(which_set='train')
    ds20_test = News20(which_set='test')
    
    #print ds20_train.X.shape
    #print ds20_test.X.shape
	
    #monitoring_dataset = {'train': ds20_train, 'test': ds20_test}
    monitoring_dataset = {'train': ds20_train}
    rbm_model = ReplicatedSoftmaxRBM(n_vis=2000, n_hid=50)

    #total_cost = MyCD_energy_scan(k=1)
    total_cost = CDk(k=1)
    
    alg = SGD(learning_rate=0.001, cost=total_cost, batch_size=100, init_momentum=0.9, monitoring_dataset=monitoring_dataset,
              termination_criterion=EpochCounter(max_epochs=1000))
    
    #MonitorBasedLRAdjuster(dataset_name='valid'),MomentumAdjustor(start=1, saturate=20, final_momentum=.99)
    train = Train(dataset=ds20_train, model=rbm_model, algorithm=alg,
            #extensions=[MonitorBasedSaveBest(channel_name='test_recon_error', save_path='my_rsm_1110.pkl')],
            save_path='my_rsm_trainsave_1110.pkl',
            save_freq=10)
    
    train.main_loop()
    

