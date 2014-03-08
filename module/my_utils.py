# -*- coding: utf-8 -*-
"""有用的函数"""

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX, as_floatX

import time
from itertools import izip
from collections import OrderedDict
import pylab


#计算矩阵每一个列向量的稀疏度，输出输出其平均值
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

#根据设计矩阵绘图
def get_visual_gray(X, shape0, shape1, save_path=None, dpi=240, random=False):
    """当X中的图片数不能填满rows*cols个位置时，仍能够正常绘图，
       shape0是绘图方阵的长和宽，shape1是一幅图片的长和宽
       支持灰度图，可以考虑分成两个函数
       带预处理
       假设X是设计矩阵"""
    sample_num = X.shape[0]
    rows, cols = shape0
    cnt = 0
    #预处理
    X = X / (numpy.sqrt((X**2).sum(axis=1))[:, numpy.newaxis])
    if random is True and sample_num > rows*cols:
        indexes = numpy.random.choice(a=xrange(sample_num), size=rows*cols, replace=False)
    else:
        random = False
    for i in xrange(rows):
        for j in xrange(cols):
            if random is False:
                Xi = X[i*cols+j]
            else:
                Xi = X[indexes[i*cols+j]]
            #绘图
            pylab.subplot(rows, cols, i*cols+j+1); pylab.axis('off'); pylab.imshow(Xi.reshape(shape1[0], shape1[1]), cmap=pylab.cm.gray)
            cnt += 1
            if cnt >= sample_num:
                pylab.show()
                if save_path is not None:
                    pylab.savefig(save_path, dpi=dpi)
                return
    if save_path is not None:
        pylab.savefig(save_path, dpi=dpi)
    pylab.show()
    return

def get_visual_rgb(X, shape0, shape1, save_path=None, dpi=240, rgb=False, random=False):
    """当X中的图片数不能填满rows*cols个位置时，仍能够正常绘图，假定X是设计矩阵
       目前仅支持分别绘制三个channel的灰度图"""
    sample_num = X.shape[0]
    rows, cols = shape0
    #对于rgb图像。默认其设计矩阵轴向是('b', 'c', 0, 1)
    X = X.reshape(X.shape[0], 3, shape1[0], shape1[1])
    Xr = X[:, 0, :, :]
    Xg = X[:, 1, :, :]
    Xb = X[:, 2, :, :]
    l = [Xr, Xg, Xb]
    for images in l:
        get_visual_gray(images, shape0=shape0, shape1=shape1, save_path=save_path, dpi=dpi, random=random)
        
    return l

#产生随机种子，可以用来初始化随机数生成器
def get_seed():
    """利用当前时间，产生一个种子，可以使用种子初始化一个随机数生成器"""
    t = [str(ord(s)) for s in str(time.clock())] #求出每个字符的asc码
    a = ''
    for i in t:
        a = a + i
    return int(a)

"""
希望具有的功能：能对指定层次的权值矩阵进行可视化
输入：一个包含每一层的权值的列表，一个包含每一层偏置的列表，指定希望可视化的隐层编号
虽然该类是针对线性+非线性的神经网络设计的，但是事实上稍加改动就可以适合任意传播函数
"""
class DeepVisual(object):
    """使用随机梯度法，计算多层神经网络的隐单元极大激活投影"""
    def __init__(self, W_list, b_list):
        """"""
        self.check(W_list, b_list) #检测参数合法性
        #将输入列表中的数值参数转化为shared符号变量
        for i in xrange(len(W_list)):
            W_list[i] = sharedX(W_list[i])
            b_list[i] = sharedX(b_list[i])
        self.W_list = W_list
        self.b_list = b_list
    
    def check(self, W_list, b_list):
        """参数合法性检测"""
        assert len(W_list) == len(b_list)
        for W, b in izip(W_list, b_list):
            assert isinstance(W, numpy.ndarray) and W.ndim == 2
            assert isinstance(b, numpy.ndarray) and W.ndim == 2
            
    def non_linear(self, x):
        """非线性过程"""
        #return T.nnet.sigmoid(x)
        return T.square(x)
        
    def get_target(self, v, n): #这里要求v是一个符号向量
        """
            使用输入的shared符号变量，计算第n个隐层的激活值(向量)的符号表达式
            v: 初始输入shared变量
            n: 所希望可视化的层数，默认最下方隐层为n=1
        """
        #data = T.matrix('data')
        f_stack = []
        f_stack.append(v)
        for i in xrange(n):
            f_stack.append(self.non_linear(T.dot(f_stack[-1], self.W_list[i]) + self.b_list[i]))
        target_vector = f_stack[-1]
        return  target_vector#返回计算结果的符号表达式
    
    def SteepestGradientDescent(self, y, x, num_epochs, lr, momentum=0.0):
        """y是一个关于x的函数，y与x都是符号变量；为了极大化y，使用最速梯度下降算法"""
        #hi = target_vector[i] #通过切片获得索引为i的隐单元的输出符号表达式
        gy = T.grad(y, x)
        updates = OrderedDict()
        inc = sharedX(x.get_value() * 0.)
        updated_inc = momentum * inc - lr * gy #本轮速度计算公式
        updates[inc] = updated_inc #更新速度记录
        updates[x] = x + updated_inc #更新参数            
        train_func = theano.function(inputs=[], outputs=[x, gy, y], updates=updates) 
        #show_func = theano.function(inputs=[], outputs=v) #输出当前位置坐标
        #g_func = theano.function([], ghi) #输出当前梯度值
        cur_x = x.get_value() #保存当前坐标，如果经过一轮下降后坐标没有变化，则终止迭代
        cnt = 0
        for e in xrange(num_epochs):
            #print 'epoch=', e
            #print 'g_func=', g_func()
            train_func()
            temp_x = x.get_value()
            temp_x = temp_x / numpy.sqrt(numpy.sum(temp_x**2))
            #终止条件判断：如果经过一轮下降+归整化，坐标没有发生变化，则终止
            if numpy.all(numpy.abs(temp_x - cur_x) < 0.00000001):
            #if numpy.all(temp_x - cur_x < 1e-10): #此语句会导致bug
                break
            cur_x = temp_x
            x.set_value(temp_x)
            cnt += 1
            #print 'show_func=', show_func()
        return x.get_value(), cnt #返回最终坐标和实际循环次数
    
    def train(self, target_layer=None, lr=0.01, num_epochs=10, init_v=None, momentum=0.0):
        """训练函数
            lr: 学习速率，optional. 默认为0.01
            W_list: list, 权值列表，元素为numpy ndarray
            b_list: list，偏置列表，元素为numpy.ndarray
            target_layer: int, optional.
                希望可视化的层的编号，第一层隐层认为是1. 如果None，则默认可视化最顶层
            num_epochs: int，optional. 
                求解最优问题时延梯度方向下降的最大轮次. 默认值为10，事实上往往一轮即收敛.
            init_v: 1d-ndarray, 起点坐标, optional, 默认为原点.
            momentum: float，动量法，optional. 默认momentum=0，不使用动量法
        """
        if target_layer is None:
            target_layer = len(self.W_list) #默认可视化最高层
        #初始化起始搜索位置init_v
        if init_v is None:
            init_v = numpy.zeros(shape=self.W_list[0].get_value().shape[0], dtype=float)
        #初始化起点v，shared化
        v = sharedX(init_v) #将v转化为shared
        
        #获得目标函数输出的符号表达式
        target_vector = self.get_target(v, target_layer)
        
        n_hid = self.W_list[-1].get_value().shape[1] #网络最顶层隐单元数，也是求解的最外层循环次数，每次循环求出一个隐单元的最大激活投影
        
        #self.vv是用来储存最终得到的n_hid个极大激活投影的模板
        self.projection = numpy.zeros(shape=(self.W_list[-1].get_value().shape[1], self.W_list[0].get_value().shape[0]), dtype=float)
        
        t_func = theano.function([], target_vector) #输出当前目标函数值!!!
        self.e_record = [] #记录每个隐单元求投影过程中的实际下降次数，
        for i in xrange(n_hid):
            hi = target_vector[i] #通过切片获得索引为i的隐单元的输出符号表达式
            self.projection[i, :], e = self.SteepestGradientDescent(lr=lr, y=hi, x=v, num_epochs=num_epochs, momentum=momentum)
            self.e_record.append(e)
            v.set_value(init_v) #重置v的值维初始值，以便计算下一个隐单元的极大激活投影
        assert len(self.e_record) == self.projection.shape[0]
        return self.projection

        

        
        
        
