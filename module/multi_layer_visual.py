#coding: utf8
"""
希望具有的功能：能对指定层次的权值矩阵进行可视化
输入：一个包含每一层的权值的列表，一个包含每一层偏置的列表
"""
import numpy
import theano
import theano.tensor as T
import pylab
import time
from itertools import izip
from pylearn2.utils import sharedX
from collections import OrderedDict


class DeepVisual(object):
    """使用随机梯度法，计算多层神经网络的隐单元极大激活投影"""
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
        for e in xrange(num_epochs):
            #print 'epoch=', e
            #print 'g_func=', g_func()
            train_func()
            temp_x = x.get_value()
            temp_x = temp_x / numpy.sqrt(numpy.sum(temp_x**2))
            if numpy.all(temp_x == cur_x): #终止条件判断
                break
            x.set_value(temp_x)
            #print 'show_func=', show_func()
        return x.get_value()
    
    def train(self, lr, W_list, b_list, num_epochs, target_layer, init_v=None, momentum=0.0):
        """训练函数
            lr: 学习速率
            W_list: list, 权值列表，元素为numpy ndarray
            b_list: list，偏置列表，元素为numpy.ndarray
            target_layer: int, 希望可视化的层的编号，第一层隐层默认为1
            num_epochs: int，求解最优问题时延梯度方向下降的最大轮次
            init_v: 1d-ndarray, 起点坐标
            momentum: float，动量法，默认momentum=0.，不使用动量法
        """
        self.check(W_list, b_list) #检测参数合法性
        
        #将输入列表中的数值参数转化为shared符号变量
        for i in xrange(len(W_list)):
            W_list[i] = sharedX(W_list[i])
            b_list[i] = sharedX(b_list[i])
        self.W_list = W_list
        self.b_list = b_list
            
        #初始化起始搜索位置init_v
        if init_v is None:
            init_v = numpy.zeros(shape=W_list[0].get_value().shape[0], dtype=float)
        #初始化起点v，shared化
        v = sharedX(init_v) #将v转化为shared
        
        #获得目标函数输出的符号表达式
        target_vector = self.get_target(v, target_layer)
        
        n_hid = W_list[-1].get_value().shape[1] #网络最顶层隐单元数，也是求解的最外层循环次数，每次循环求出一个隐单元的最大激活投影
        
        #self.vv是用来储存最终得到的n_hid个极大激活投影的模板
        self.projection = numpy.zeros(shape=(W_list[-1].get_value().shape[1], W_list[0].get_value().shape[0]), dtype=float)
        
        t_func = theano.function([], target_vector) #输出当前目标函数值!!!
        for i in xrange(n_hid):
            hi = target_vector[i] #通过切片获得索引为i的隐单元的输出符号表达式
            self.projection[i, :] = self.SteepestGradientDescent(lr=lr, y=hi, x=v, num_epochs=num_epochs, momentum=momentum)
            v.set_value(init_v) #重置v的值维初始值，以便计算下一个隐单元的极大激活投影

        
