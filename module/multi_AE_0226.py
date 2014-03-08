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

#1.基本型AutoEncoder

#1.1.基本型AE的相关Cost
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

#1.2.基本型AE模型
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
    
    def get_train_enc(self, data):
        """训练时编码，为例如DAE这类模型服务"""
        v = data
        if self.corruptor is not None: 
            v = self.corruptor(v)
        mid = self.get_enc(v)
        return mid
    
    def get_final_output(self, data):
        """计算由输入得到的最终输出"""
        v = data
        if self.corruptor is not None: 
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
            cost = SumOfCosts(costs=[cost,  self.extra_cost])
        return cost
   
#2.发展型双模态(文本+图像)AE

#2.1.双模态AE的四个Cost
class CorreCost(Cost):
    """2014.03.07修正：加入额外的两种距离函数下的corre_cost表达式"""
    def __init__(self, img_dim, txt_dim, dist_func='l1'):
        """"""
        #self.__dict__.update(locals())
        #del self.self
        assert dist_func in ['cos', 'euc', 'l1']
        self.dist_func = dist_func
        self.img_dim = img_dim
        self.txt_dim = txt_dim
    
    def expr(self, model, data):
        """代价函数表达式"""
        img_data = data[:, 0: self.img_dim]
        txt_data = data[:, self.img_dim: self.img_dim + self.txt_dim]
        code_img = model.img_AE.get_enc(img_data)
        code_txt = model.txt_AE.get_enc(txt_data)
        
        if self.dist_func == 'l1':
            C_cost = T.mean(T.abs_(code_img - code_txt).sum(axis=1)) #编码维数相同由model保证
        if self.dist_func == 'cos':
            code_img_norm = numpy.sqrt(numpy.sum(code_img**2, axis=1))
            code_txt_norm = numpy.sqrt(numpy.sum(code_txt**2, axis=1))
            #极大化余弦相似度即极小化负余弦相似度
            C_cost = - T.mean( ( (code_img/code_img_norm[:, numpy.newaxis]) * (code_txt/code_txt_norm[:, numpy.newaxis]) ).sum(axis=1))
        if self.dist_func == 'euc':
            C_cost = T.mean( T.sqrt( T.square(code_img - code_txt).sum(axis=1) ) ) 

        return C_cost
    
    def get_data_specs(self, model):
        """"""
        return model.get_monitoring_data_specs()
    
class CombineCrossEntropyCost(Cost):
    """图像端恢复图像端输入，文本端恢复文本端输入"""
    def __init__(self, img_dim, txt_dim):
        """"""
        self.img_dim = img_dim
        self.txt_dim = txt_dim
    
    def expr(self, model, data):
        """代价函数表达式"""
        img_data = data[:, 0: self.img_dim]
        txt_data = data[:, self.img_dim: self.img_dim + self.txt_dim]
        output_img = model.img_AE.get_final_output(img_data)
        output_txt = model.txt_AE.get_final_output(txt_data)
        
        cross_entropy_img = -(img_data * T.log(output_img) + (1. - img_data) * T.log(1. - output_img)).sum(axis=1).mean()
        cross_entropy_txt = -(txt_data * T.log(output_txt) + (1. - txt_data) * T.log(1. - output_txt)).sum(axis=1).mean()

        return cross_entropy_img + cross_entropy_txt
    
    def get_data_specs(self, model):
        """"""
        return model.get_monitoring_data_specs()
    
class CrossModalCrossEntropyCost(Cost):
    """图像端恢复文本端输入，文本端恢复图像端输入"""
    def __init__(self, img_dim, txt_dim):
        """"""
        self.img_dim = img_dim
        self.txt_dim = txt_dim
    
    def expr(self, model, data):
        """代价函数表达式"""
        img_data = data[:, 0: self.img_dim]
        txt_data = data[:, self.img_dim: self.img_dim + self.txt_dim]
        
        code_img = model.img_AE.get_train_enc(img_data)
        code_txt = model.txt_AE.get_train_enc(txt_data)
        img_output_txt = model.txt_AE.get_dec(code_img)
        txt_output_img = model.img_AE.get_dec(code_txt)
        
        #注意图像端输出是文本端的重构，文本端输出是图像端重构；因此文本端输入与图像端输出比较，图像端输入与文本端输出比较
        cross_entropy_txt2img = -(img_data * T.log(txt_output_img) + (1. - img_data) * T.log(1. - txt_output_img)).sum(axis=1).mean()
        cross_entropy_img2txt = -(txt_data * T.log(img_output_txt) + (1. - txt_data) * T.log(1. - img_output_txt)).sum(axis=1).mean()

        return cross_entropy_txt2img + cross_entropy_img2txt

    def get_data_specs(self, model):
        """"""
        return model.get_monitoring_data_specs()
    
class FullModalCrossEntropyCost(Cost):
    """图像端同时恢复文本端输入和图像端输入，文本端同时恢复文本端输入和图像端输入"""
    def __init__(self, img_dim, txt_dim):
        """"""
        self.img_dim = img_dim
        self.txt_dim = txt_dim
    
    def expr(self, model, data):
        """代价函数表达式"""
        img_data = data[:, 0: self.img_dim]
        txt_data = data[:, self.img_dim: self.img_dim + self.txt_dim]
        
        code_img = model.img_AE.get_train_enc(img_data)
        code_txt = model.txt_AE.get_train_enc(txt_data)
        img_output_img = model.img_AE.get_dec(code_img) #图像恢复图像
        img_output_txt = model.txt_AE.get_dec(code_img) #图像恢复文本
        txt_output_txt = model.txt_AE.get_dec(code_txt) #文本恢复文本
        txt_output_img = model.img_AE.get_dec(code_txt) #文本恢复图像
        
        #注意图像端输出是文本端的重构，文本端输出是图像端重构；因此文本端输入与图像端输出比较，图像端输入与文本端输出比较
        #从文本端重构图像的cost
        cross_entropy_txt2img = -(img_data * T.log(txt_output_img) + (1. - img_data) * T.log(1. - txt_output_img)).sum(axis=1).mean()
        #从文本端重构文本的cost
        cross_entropy_txt2txt = -(txt_data * T.log(txt_output_txt) + (1. - txt_data) * T.log(1. - txt_output_txt)).sum(axis=1).mean()
        #从图像端重构文本的cost
        cross_entropy_img2txt = -(txt_data * T.log(img_output_txt) + (1. - txt_data) * T.log(1. - img_output_txt)).sum(axis=1).mean()
        #从图像端重构图像的cost
        cross_entropy_img2img = -(img_data * T.log(img_output_img) + (1. - img_data) * T.log(1. - img_output_img)).sum(axis=1).mean()

        return cross_entropy_img2img + cross_entropy_img2txt + cross_entropy_txt2txt + cross_entropy_txt2img
    
    def get_data_specs(self, model):
        """"""
        return model.get_monitoring_data_specs()

#02.25新增cost，专维AdjustableMultimodalAutoEncoder准备
class AdjustableCombineCrossEntropyCost(Cost):
    """图像端恢复图像端输入，文本端恢复文本端输入"""
    def __init__(self, img_dim, txt_dim):
        """"""
        self.img_dim = img_dim
        self.txt_dim = txt_dim
    
    def expr(self, model, data):
        """代价函数表达式"""
        img_data = data[:, 0: self.img_dim]
        txt_data = data[:, self.img_dim: self.img_dim + self.txt_dim]
        output_img = model.img_AE.get_final_output(img_data)
        output_txt = model.txt_AE.get_final_output(txt_data)
        
        cross_entropy_img = -(img_data * T.log(output_img) + (1. - img_data) * T.log(1. - output_img)).sum(axis=1).mean()
        cross_entropy_txt = -(txt_data * T.log(output_txt) + (1. - txt_data) * T.log(1. - output_txt)).sum(axis=1).mean()

        return model.alpha * cross_entropy_img + model.beta * cross_entropy_txt
    
    def get_data_specs(self, model):
        """"""
        return model.get_monitoring_data_specs()

#2.2发展型AE模型

class MyMultimodalAutoEncoder(Model, Block):
    """AutoEncoder(AE)"""
    def __init__(self, model_type=None, alpha=0.2, 
                n_vis_img=None, n_vis_txt=None, n_hid_img=None, n_hid_txt=None, corruptor_img=None, corruptor_txt=None, 
                W_img=None, W_txt=None, b_enc_img=None, b_enc_txt=None, b_dec_img=None, b_dec_txt=None, dec_f_img=True, dec_f_txt=True, 
                img_AE=None, txt_AE=None, corre_func='l1', numpy_rng=None, theano_rng=None):
        """
        model_type: String, 选择模型类型，目的是为了控制get_default_cost()方法找到所希望的训练代价
                    可选参数: 'Combine', 'CrossModal', 'FullModal'
        param: alpha, 标准代价和关联代价的权重稀疏，alpha越大则标准代价在总的代价函数中的比重越大
        param: img_AE, 图像端用AE
        param: txt_AE, 文本端用AE
        """
        Model.__init__(self) # self.names_to_del = set(); self._test_batch_size = 2
        Block.__init__(self) # self.fn = None; self.cpu_only = False
        assert model_type in ['Combine', 'CrossModal', 'FullModal']
        self.model_type = model_type
        self.alpha = alpha
        
        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(seed=19900418)
        self.numpy_rng = numpy_rng

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng
        
        #两个AE共用的部分只有随机数发生器
        if img_AE is None:
            assert n_vis_img is not None
            assert n_hid_img is not None
            
            img_AE = MyAutoEncoder(n_vis=n_vis_img, n_hid=n_hid_img, corruptor=corruptor_img, 
                    W=W_img, b_enc=b_enc_img, b_dec=b_dec_img, dec_f=dec_f_img, 
                    numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
            
        if txt_AE is None:
            assert n_vis_txt is not None
            assert n_hid_txt is not None
            txt_AE = MyAutoEncoder(n_vis=n_vis_txt, n_hid=n_hid_txt, corruptor=corruptor_txt, 
                    W=W_txt, b_enc=b_enc_txt, b_dec=b_dec_txt, dec_f=dec_f_txt, 
                    numpy_rng=self.numpy_rng, theano_rng=self.theano_rng)
        
        assert img_AE.n_hid == txt_AE.n_hid #目前的模型只能接受两端具有相同维度的编码空间
        
        self.img_AE = img_AE
        self.txt_AE = txt_AE
        
        self.W_img = img_AE.W #not used
        self.W_txt = txt_AE.W #not used
        self.b_enc_img = img_AE.b_enc #not used
        self.b_dec_img = img_AE.b_dec #not used
        self.b_enc_txt = txt_AE.b_enc #not used
        self.b_dec_txt = txt_AE.b_dec #not used
        
        self.n_vis_img = self.img_AE.n_vis
        self.n_vis_txt = self.txt_AE.n_vis
        self.n_hid_img = self.img_AE.n_hid
        self.n_hid_txt = self.txt_AE.n_hid
        self.n_vis = self.img_AE.n_vis + self.txt_AE.n_vis
        self.n_hid = self.img_AE.n_hid + self.txt_AE.n_hid
        
        self.input_space = VectorSpace(dim=self.n_vis) # add input_space
        self.output_space = VectorSpace(dim=self.n_hid) # add output_space

        #init_W = numpy.concatenate([self.img_AE.W, self.txt_AE_W], axis=1)
        #self.W = theano.shared(value=init_W, name='W', borrow=True)
        
        #参数顺序：图像权值矩阵， 图像编码偏置，图像解码偏置，文本权值矩阵，文本编码偏置，文本解码偏置
        self._params = [self.img_AE.W, self.img_AE.b_enc, self.img_AE.b_dec, self.txt_AE.W, self.txt_AE.b_enc, self.txt_AE.b_dec]  
        
    def get_enc_img(self, img_data): #not used
        """图像编码方法，线性变换+非线性变换"""
        enc_img = self.img_AE.get_enc(img_data)
        return enc_img
    
    def get_enc_txt(self, txt_data): #not used
        """文本编码方法，线性变换+非线性变换"""
        enc_img = self.txt_AE.get_enc(txt_data)
        return enc_img
        
    def get_dec_img(self, img_code): #not used
        """图像解码方法，线性变换+非线性变换"""
        dec_img = self.img_AE.get_dec(img_code)
        return dec_img
    
    def get_dec_txt(self, txt_code): #not used
        """文本解码方法，线性变换+非线性变换"""
        dec_txt = self.txt_AE.get_dec(txt_code)
        return dec_txt
    
    def get_train_enc_img(self, img_data): #not used
        """计算图像端由输入得到的最终输出，只在训练时使用"""
        train_code_img = self.img_AE.get_final_output(img_data)
        return train_code_img
    
    def get_train_enc_txt(self, txt_data): #not used
        """计算文本端由输入得到的最终输出，只在训练时使用"""
        train_code_txt = self.img_AE.get_train_enc(txt_data)
        return train_code_txt
    
    def get_monitoring_data_specs(self):
        """"""
        return (self.get_input_space(), self.get_input_source())
    
    def get_monitoring_channels(self, data):
        """"""
        img_data = data[:, 0: self.n_vis_img]
        txt_data = data[:, self.n_vis_img: self.n_vis]
        channels = {}
        
        code_img = self.img_AE.get_train_enc(img_data) #图像端编码
        code_txt = self.txt_AE.get_train_enc(txt_data) #文本端编码
        img2img = self.img_AE.get_dec(code_img) #图像恢复图像
        img2txt = self.txt_AE.get_dec(code_img) #图像恢复文本
        txt2txt = self.txt_AE.get_dec(code_txt) #文本恢复文本
        txt2img = self.img_AE.get_dec(code_txt) #文本恢复图像

        #self.model_type: 'Combine', 'FullModal'
        channel_name = 'img2img_recon_error'
        img2img_recon_error = (T.square(img2img - img_data).sum(axis=1)).mean()
        channels[channel_name] = img2img_recon_error
        #self.model_type: 'Combine', 'FullModal'
        channel_name = 'txt2txt_recon_error'
        txt2txt_recon_error = (T.square(txt2txt - txt_data).sum(axis=1)).mean()
        channels[channel_name] = txt2txt_recon_error
        #self.model_type: 'CrossModal', 'FullModal'
        channel_name = 'img2txt_recon_error'
        img2txt_recon_error = (T.square(img2txt - txt_data).sum(axis=1)).mean()
        channels[channel_name] = img2txt_recon_error
        #self.model_type: 'CrossModal', 'FullModal'
        channel_name = 'txt2img_recon_error'
        txt2img_recon_error = (T.square(txt2img - img_data).sum(axis=1)).mean()
        channels[channel_name] = txt2img_recon_error
        
        #channel_name = 'hidden_sparseness' #隐层表示稀疏度
        #h = self.get_enc(v)
        #sparseness_vector = (T.sqrt(self.n_hid) - T.sum(T.abs_(h), axis=1) / T.sqrt(T.sum(h**2, axis=1))) / (T.sqrt(self.n_hid) - 1.)
        #assert sparseness_vector.ndim == 1
        #channels[channel_name] = T.mean(sparseness_vector)
        
        #channel_name = 'W_sparseness' #权值矩阵稀疏度
        #W = self.W.T
        #sparseness_vector = (T.sqrt(self.n_vis) - T.sum(T.abs_(W), axis=1) / T.sqrt(T.sum(W**2, axis=1))) / (T.sqrt(self.n_vis) - 1.)
        #assert sparseness_vector.ndim == 1
        #channels[channel_name] = T.mean(sparseness_vector)

        return channels
    
    def get_default_cost(self):
        """Based on Salah Rifai's ICML11's Paper"""
        if self.model_type is 'Combine':
            pro_cost = CombineCrossEntropyCost(self.n_vis_img, self.n_vis_txt)
        elif self.model_type is 'CrossModal':
            pro_cost = CrossModalCrossEntropyCost(self.n_vis_img, self.n_vis_txt)
        else:
            pro_cost = FullModalCrossEntropyCost(self.n_vis_img, self.n_vis_txt)
        
        corre_cost = CorreCost(self.n_vis_img, self.n_vis_txt)
        
        cost = SumOfCosts(costs=[(self.alpha, pro_cost), (1. - self.alpha, corre_cost)])
        #SumOfCosts(costs=[MyCD_energy_scan(k=1), (1/ 0.05, HonglakLeeSparse(p=0.05))])
        return cost

#3.新版双参数combineAE，出发点是两个端的AE代价可以用参数进行调整
class AdjustableMultimodalAutoEncoder(MyMultimodalAutoEncoder):
    """可调整的multiAE，继承MyMultimodalAutoEncoder"""
    def __init__(self, model_type=None, alpha=0.5, beta=0.5,
                n_vis_img=None, n_vis_txt=None, n_hid_img=None, n_hid_txt=None, corruptor_img=None, corruptor_txt=None, 
                W_img=None, W_txt=None, b_enc_img=None, b_enc_txt=None, b_dec_img=None, b_dec_txt=None, dec_f_img=True, dec_f_txt=True, 
                img_AE=None, txt_AE=None, corre_func='l1', numpy_rng=None, theano_rng=None):
        """这里新增了参数beta，而alpha在self.get_default_cost中的作用也与原来不同"""
        super(AdjustableMultimodalAutoEncoder, self).__init__(model_type=model_type, alpha=alpha, 
                n_vis_img=n_vis_img, n_vis_txt=n_vis_txt, n_hid_img=n_hid_img, n_hid_txt=n_hid_txt, 
                corruptor_img=corruptor_img, corruptor_txt=corruptor_txt, W_img=W_img, W_txt=W_txt, 
                b_enc_img=b_enc_img, b_enc_txt=b_enc_txt, b_dec_img=b_dec_img, b_dec_txt=b_dec_txt, dec_f_img=dec_f_img, dec_f_txt=dec_f_txt, 
                img_AE=img_AE, txt_AE=txt_AE, corre_func=corre_func, numpy_rng=numpy_rng, theano_rng=theano_rng)
            
        self.beta = beta
        
    def get_default_cost(self):
        """默认代价函数"""
        pro_cost = AdjustableCombineCrossEntropyCost(self.n_vis_img, self.n_vis_txt) #初始化一个代价对象
        
        corre_cost = CorreCost(self.n_vis_img, self.n_vis_txt)
        cost = SumOfCosts(costs=[pro_cost, corre_cost])
        #SumOfCosts(costs=[MyCD_energy_scan(k=1), (1/ 0.05, HonglakLeeSparse(p=0.05))])
        return cost

    def get_monitoring_channels(self, data):
        """由于combie型"""
        img_data = data[:, 0: self.n_vis_img]
        txt_data = data[:, self.n_vis_img: self.n_vis]
        channels = {}
        
        code_img = self.img_AE.get_train_enc(img_data) #图像端编码
        code_txt = self.txt_AE.get_train_enc(txt_data) #文本端编码
        img2img = self.img_AE.get_dec(code_img) #图像恢复图像
        #img2txt = self.txt_AE.get_dec(code_img) #图像恢复文本
        txt2txt = self.txt_AE.get_dec(code_txt) #文本恢复文本
        #txt2img = self.img_AE.get_dec(code_txt) #文本恢复图像

        #self.model_type: 'Combine', 'FullModal'
        channel_name = 'img2img_recon_error'
        img2img_recon_error = (T.square(img2img - img_data).sum(axis=1)).mean()
        channels[channel_name] = img2img_recon_error
        #self.model_type: 'Combine', 'FullModal'
        channel_name = 'txt2txt_recon_error'
        txt2txt_recon_error = (T.square(txt2txt - txt_data).sum(axis=1)).mean()
        channels[channel_name] = txt2txt_recon_error

        return channels

#4.训练用例
#4.1.发展型AE的训练示例
#这里没有使用双模态数据集，而是将mnist数据集看作一个由两个各有392维的数据集拼接而成的数据集，前392维特征作为图像特征，后392维特征作为文本特征
if __name__ == '__main__':
    
    from pylearn2.datasets.mnist import MNIST
    from pylearn2.training_algorithms.sgd import SGD
    from pylearn2.costs.cost import SumOfCosts
    from pylearn2.train import Train
    from pylearn2.termination_criteria import MonitorBased
    from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
    from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
    from pylearn2.training_algorithms.sgd import MomentumAdjustor
    from pylearn2.termination_criteria import EpochCounter
    
    from module.dataset_from_design import DatasetFromDesign
    
    import cPickle
    #from pylearn2.datasets.preprocessing import GlobalContrastNormalization, ZCA, ExtractPatches, Pipeline
    
    dsm_train = MNIST(which_set='train', start=0, stop=500, one_hot=True)
    dsm_valid = MNIST(which_set='train', start=500, stop=600, one_hot=True)
    dsm_test = MNIST(which_set='test', start=0, stop=100, one_hot=True)
    #f = open('/home/zanghu/Pro_Datasets/wikipedia_article_combine_pylearn2/wpa_combine_pylearn2_train.pkl')
    #dswpa_train = cPickle.laod(f)
    #f.close()
    
    #f = open('/home/zanghu/Pro_Datasets/wikipedia_article_combine_pylearn2/wpa_combine_pylearn2_test.pkl')
    #dswpa_test = cPickle.laod(f)
    #f.close()
	
    monitoring_dataset = {'train': dsm_train, 'valid': dsm_valid, 'test': dsm_test}
    #monitoring_dataset = {'train': dsm_train}
	
    ae_model = MyMultimodalAutoEncoder(model_type='Combine', alpha=0.2, n_vis_img=392, n_vis_txt=392, n_hid_img=100, n_hid_txt=100)
    
    alg = SGD(learning_rate=0.1, cost=None, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset,
              termination_criterion=EpochCounter(max_epochs=15))
    
    train = Train(dataset=dsm_train, model=ae_model, algorithm=alg, save_path='ae_save.pkl', save_freq=5)
    
    train.main_loop()

#4.2.新版AdjustedMultimodalAutoEncoder训练用例
if __name__ == '__main__':
    
    from pylearn2.datasets.mnist import MNIST
    from pylearn2.training_algorithms.sgd import SGD
    from pylearn2.costs.cost import SumOfCosts
    from pylearn2.train import Train
    from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
    from pylearn2.training_algorithms.sgd import MomentumAdjustor
    from pylearn2.termination_criteria import EpochCounter
    
    from module.dataset_from_design import DatasetFromDesign
    
    import cPickle
    #from pylearn2.datasets.preprocessing import GlobalContrastNormalization, ZCA, ExtractPatches, Pipeline
    
    dsm_train = MNIST(which_set='train', start=0, stop=500, one_hot=True)
    dsm_valid = MNIST(which_set='train', start=500, stop=600, one_hot=True)
    dsm_test = MNIST(which_set='test', start=0, stop=100, one_hot=True)
	
    monitoring_dataset = {'train': dsm_train, 'valid': dsm_valid, 'test': dsm_test}
    #monitoring_dataset = {'train': dsm_train}
	
    ae_model = AdjustableMultimodalAutoEncoder(model_type='Combine', alpha=0.5, beta=0.5, n_vis_img=392, n_vis_txt=392, n_hid_img=100, n_hid_txt=100)
    
    alg = SGD(learning_rate=0.1, cost=None, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset,
              termination_criterion=EpochCounter(max_epochs=15))
    
    train = Train(dataset=dsm_train, model=ae_model, algorithm=alg, save_path='ae_save.pkl', save_freq=5)
    
    train.main_loop()

