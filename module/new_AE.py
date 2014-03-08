#coding: utf8
import theano
import theano.tensor as T
import numpy
from theano.tensor.shared_randomstreams import RandomStreams

from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.costs.cost import Cost, SumOfCosts
from pylearn2.space import VectorSpace
from pylearn2.corruption import BinomialCorruptor, GaussianCorruptor
from pylearn2.utils import sharedX, as_floatX

theano.config.compute_test_value = 'off'

def get_sparseness(W, n):
    #if isinstance(W, theano.tensor.sharedvar.TensorSharedVariable):
    #n = numpy.prod(W.shape)
    sparseness = (T.sqrt(n) - T.abs_(W).sum() / T.sqrt(T.square(W).sum())) / (T.sqrt(n) -1)
    return sparseness

class MSECost(Cost):
    
    #def __init__(self):
    #    pass
    
    def get_data_specs(self, model):
        
        return model.get_monitoring_data_specs()
        
    def expr(self, model, data):
        
        v = data
        output = model.get_final_output(v)
        recon_error = (T.square(output - v).sum(axis=1)).mean()
        return 0.5 * recon_error
    
class CrossEntropyCost(Cost):
    
    def expr(self, model, data):
        v = data
        z = model.get_final_output(v)
        # x_i maybe take zero value,but x is not in log, z is directly derived from model.nonlinear(), its value must between (0, 1)
        cross_entropy = -T.sum(v * T.log(z) + (1. - v) * T.log(1. - z), axis=1)
        return cross_entropy.mean()

    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()
    
class L2WeightDecay(Cost):
    
    def expr(self, model, data):
        
        weight_decay = T.square(model.W1).sum() + T.square(model.W2).sum()
        return weight_decay
    
    def get_data_specs(self, model):
        
        return model.get_monitoring_data_specs()
    
class L1WeightDecay(Cost):
    
    def expr(self, model, data):
        
        weight_decay = T.abs_(model.W).sum()
        return weight_decay
    
    def get_data_specs(self, model):
        
        return model.get_monitoring_data_specs()
    
    def get_monitoring_channels(self, model, data, **kwargs):
        
        channels = OrderedDict()
        n = model.n_vis * model.n_hid
        channels['weight_sparseness'] = get_sparseness(model.W, n)
        return channels
    
class L1Sparse(Cost):
    
    def expr(self, model, data):
        
        v = data
        mid = model.get_enc(v)
        L1_sparse = T.abs_(mid).sum(axis=1).mean()
        return L1_sparse
    
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()
    
class RLeSparse(Cost): # Ranzato-LeCun 2007
    
    def expr(self, model, data):
        
        v = data
        z = (model.get_enc(v)) ** 2
        h = T.log(1 + T.square(z)).sum(axis=1).mean()
        return h
    
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()
    
class CS294Sparse(Cost):
    
    def __init__(self, rou=0.05):
        self.rou = rou
    
    def expr(self, model, data):
        
        v = data
        mid = model.get_enc(v)
        rou_mid = mid.mean(axis=0)
        cs294_sparse = (self.rou * T.log(self.rou / rou_mid) + (1 - self.rou) * T.log((1 - self.rou) / (1 - rou_mid))).sum()
        return cs294_sparse
    
    def get_data_specs(self, model):
        """"""
        return model.get_monitoring_data_specs()
    
class MyAutoEncoder(Model, Block):
    """AutoEncoder(AE)"""
    def __init__(self, n_vis, n_hid, corruptor=None, W=None, b_enc=None, b_dec=None, numpy_rng=None, dec_f=True, extra_cost=None,theano_rng=None):
        """构造函数
        dec_f: 解码单元是否包含非线性函数
        extra_cost:除了基本的MSE Cost和CE Cost之外的代价函数其他惩罚项，例如稀疏惩罚，weight decay等等. 
                用于self.get_default_cost()方法中. 这样依赖需要在模型初始化之前加入希望添加的惩罚项即可.
        """
        Model.__init__(self) # self.names_to_del = set(); self._test_batch_size = 2
        Block.__init__(self) # self.fn = None; self.cpu_only = False

        self.n_vis = n_vis
        self.n_hid = n_hid
        self.extra_cost = extra_cost
        self.dec_f = dec_f
        
        if corruptor is not None:
            self.corruptor = corruptor
        
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
                      low= -4 * numpy.sqrt(6. / (n_hid + n_vis + 1.)),
                      high= 4 * numpy.sqrt(6. / (n_hid + n_vis + 1.)),
                      size=(n_vis, n_hid)),
                      dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=init_W, name='W', borrow=True)

        if b_enc is None:
            # create shared variable for hidden units bias
            b_enc = theano.shared(value=numpy.zeros(n_hid, dtype=theano.config.floatX), name='b_enc', borrow=True)

        if b_dec is None:
            # create shared variable for visible units bias
            b_dec = theano.shared(value=numpy.zeros(n_vis, dtype=theano.config.floatX), name='b_dec', borrow=True)

        self.W = W
        self.b_enc = b_enc
        self.b_dec = b_dec

        self._params = [self.W, self.b_enc, self.b_dec]  
        
    def get_enc(self, data):
        """编码方法，线性变换+非线性变换"""
        v = data
        linear_enc = T.dot(v, self.W) + self.b_enc
        enc = self.nonlinear(linear_enc)
        return enc
        
    def get_dec(self, z):
        """解码方法，线性变换+非线性变换"""
        linear_dec = T.dot(z, (self.W).T) + self.b_dec
        if self.dec_f is True: #如果初始化模型时要求decoder包含非线性变换
            dec = self.nonlinear(linear_dec)
        else:
            dec = linear_dec
        return dec
        
    def nonlinear(self, x):
        
        return T.nnet.sigmoid(x)
    
    def get_final_output(self, data):
        """计算由输入得到的最终输出"""
        v = data
        if hasattr(self, 'corruptor'): 
            v = self.corruptor(v)
        mid = self.get_enc(v)
        output = self.get_dec(mid)
        return output
    
    def get_monitoring_data_specs(self):
        """"""
        return (self.get_input_space(), self.get_input_source())
    
    def get_monitoring_channels(self, data):
        """"""
        v = data
        channels = {}
        
        channel_name = 'recon_error'
        output = self.get_final_output(v)
        recon_error = (T.square(output - v).sum(axis=1)).mean()
        channels[channel_name] = recon_error
        
        channel_name = 'hidden_sparseness' #隐层表示稀疏度
        h = self.get_enc(v)
        sparseness_vector = (T.sqrt(self.n_hid) - T.sum(T.abs_(h), axis=1) / T.sqrt(T.sum(h**2, axis=1))) / (T.sqrt(self.n_hid) - 1.)
        assert sparseness_vector.ndim == 1
        channels[channel_name] = T.mean(sparseness_vector)
        
        channel_name = 'W_sparseness' #权值矩阵稀疏度
        W = self.W.T
        sparseness_vector = (T.sqrt(self.n_vis) - T.sum(T.abs_(W), axis=1) / T.sqrt(T.sum(W**2, axis=1))) / (T.sqrt(self.n_vis) - 1.)
        assert sparseness_vector.ndim == 1
        channels[channel_name] = T.mean(sparseness_vector)
        

        return channels
    
    def get_default_cost(self):
        """Based on Salah Rifai's ICML11's Paper"""
        if self.dec_f is True:
            cost = CrossEntropyCost()
        else:
            cost = MSECost()
            
        if self.extra_cost is not None:
            cost = SumOfCosts(costs=[MyCD_energy_scan(k=1), self.extra_cost])
        return cost
    
