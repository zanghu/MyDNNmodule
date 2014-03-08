#coding: utf8
import theano
import theano.tensor as T
import numpy
from sklearn.linear_model import LogisticRegression
from module.multi_AE import MyMultimodalAutoEncoder
from module.dataset_from_design import DatasetFromDesign
from module.new_query import p_r
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.cost import SumOfCosts
from pylearn2.train import Train
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
from pylearn2.training_algorithms.sgd import MomentumAdjustor
from pylearn2.termination_criteria import EpochCounter
import cPickle
import time
import os
import sys

def make_dataset(X_img_train, X_txt_train, y_train, X_img_test, X_txt_test, y_test):
    """接受各个模态的输入设计矩阵，以及训练集和测试集类标"""
    assert isinstance(X_img_test, numpy.ndarray) and isinstance(X_img_train, numpy.ndarray), X_img_test.ndim == 2 and X_img_train.ndim == 2
    assert isinstance(X_txt_test, numpy.ndarray) and isinstance(X_txt_train, numpy.ndarray), X_txt_test.ndim == 2 and X_txt_train.ndim == 2
    assert isinstance(y_test, numpy.ndarray) and isinstance(y_train, numpy.ndarray), y_test.ndim == 1 and y_train.ndim == 1
    
    X_train = numpy.concatenate([X_img_train, X_txt_train], axis=1)
    X_test = numpy.concatenate([X_img_test, X_txt_test], axis=1)
    #print 'in make_dataset, X_train: ', X_train.shape
    #print 'in make_dataset, X_test: ', X_test.shape

    dsit_train = DatasetFromDesign(design_matrix=X_train, labels=y_train)
    dsit_test = DatasetFromDesign(design_matrix=X_test, labels=y_test)
    
    return [dsit_train, dsit_test]

def propup_design_matrix(X_train, X_test, ae_model):
    """参数为作为模型输入的设计矩阵，返回模型输出的设计矩阵"""
    #检测模型类型，本函数的ae_model参数直接收该类对象
    assert isinstance(ae_model, MyMultimodalAutoEncoder) 
    assert isinstance(X_test, numpy.ndarray) and isinstance(X_train, numpy.ndarray), X_test.ndim == 2 and X_train.ndim == 2
    
    img_units = ae_model.n_vis_img #图像端输入单元数
    txt_units = ae_model.n_vis_txt #文本端输入单元数
    total_units = img_units + txt_units 
    
    X_img_train = X_train[:, 0: img_units]
    X_txt_train = X_train[:, img_units: total_units]
    X_img_test = X_test[:, 0: img_units]
    X_txt_test = X_test[:, img_units: total_units]

    x = T.matrix()
    f_img = theano.function([x], ae_model.img_AE.get_enc(x))
    f_txt = theano.function([x], ae_model.txt_AE.get_enc(x))

    X_img_propup_train = f_img(numpy.cast['float32'](X_img_train))
    X_txt_propup_train = f_txt(numpy.cast['float32'](X_txt_train))
    X_propup_train = numpy.concatenate([X_img_propup_train, X_txt_propup_train], axis=1)

    X_img_propup_test = f_img(numpy.cast['float32'](X_img_test))
    X_txt_propup_test = f_txt(numpy.cast['float32'](X_txt_test))
    X_propup_test = numpy.concatenate([X_img_propup_test, X_txt_propup_test], axis=1)
    
    return [X_img_propup_train, X_txt_propup_train, X_img_propup_test, X_txt_propup_test, X_propup_train, X_propup_test]

def model_evaluate(X_img_train, X_txt_train, y_train, X_img_test, X_txt_test, y_test, layer_num='', prefix='', suffix='', save_path=''):
    """封装成一个函数， 参数为：本层输入的训练集设计矩阵，测试集设计矩阵，经过训练后的模型输出的结果"""
    #检参数检验
    assert isinstance(X_img_test, numpy.ndarray) and isinstance(X_img_train, numpy.ndarray), X_img_test.ndim == 2 and X_img_train.ndim == 2
    assert isinstance(X_txt_test, numpy.ndarray) and isinstance(X_txt_train, numpy.ndarray), X_txt_test.ndim == 2 and X_txt_train.ndim == 2
    assert isinstance(y_test, numpy.ndarray) and isinstance(y_train, numpy.ndarray), y_test.ndim == 1 and y_train.ndim == 1

    X_train = numpy.concatenate([X_img_train, X_txt_train], axis=1)
    X_test = numpy.concatenate([X_img_test, X_txt_test], axis=1)
    
    #分类测试
    print '测试模型分类性能+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    clf  = LogisticRegression()
    clf.fit(X_train, y_train)
    #print 'use img + txt:'
    print 'use img + txt on train-dataset: ', clf.score(X_train, y_train)
    print 'use img + txt on test-dataset: ', clf.score(X_test, y_test)

    clfi  = LogisticRegression()
    clfi.fit(X_img_train, y_train)
    #print 'use img only:'
    print 'use img only on train-dataset: ', clfi.score(X_img_train, y_train)
    print 'use img only on test-dataset: ', clfi.score(X_img_test, y_test)

    clft  = LogisticRegression()
    clft.fit(X_txt_train, y_train)
    #print 'use txt only:'
    print 'use txt only on train-dataset: ', clft.score(X_txt_train, y_train)
    print 'use txt only on test-dataset: ', clft.score(X_txt_test, y_test)

    #跨模态检索测试，以检索到与query同类的数据为正确
    
    print '使用普通余弦相似度检索++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print '1.1测试集文本检索测试集图像:'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2test_txt2img' + suffix, X_train=X_img_test, label_train=y_test, X_test=X_txt_test, label_test=y_test, pic=True, dist_func='cos', prefix='layer' + str(layer_num) + '_test2test_' + prefix, suffix='_txt2img_cos_' + suffix, save_results=False, save_path=save_path+'cos')
    print '1.2测试集图像检索测试及文本:'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2test_img2txt' + suffix, X_train=X_txt_test, label_train=y_test, X_test=X_img_test, label_test=y_test, pic=True, dist_func='cos', prefix='layer' + str(layer_num) + '_test2test_' + prefix, suffix='_img2txt_cos_' + suffix, save_results=False, save_path=save_path+'cos')

    print '2.1测试集文本检索训练集图像:'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2train_txt2img' + suffix, X_train=X_img_train, label_train=y_train, X_test=X_txt_test, label_test=y_test, pic=True, dist_func='cos', prefix='layer' + str(layer_num) + '_test2train_' + prefix, suffix='_txt2img_cos_' + suffix, save_results=False, save_path=save_path+'cos')
    print '2.2测试集图像检索训练集文本'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2train_img2txt' + suffix, X_train=X_txt_train, label_train=y_train, X_test=X_img_test, label_test=y_test, pic=True, dist_func='cos', prefix='layer' + str(layer_num) + '_test2train_' + prefix, suffix='_img2txt_cos_' + suffix, save_results=False, save_path=save_path+'cos')

    print '使用调整后余弦相似度检索+++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print '3.1测试集文本检索测试集图像:'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2test_txt2img' + suffix, X_train=X_img_test, label_train=y_test, X_test=X_txt_test, label_test=y_test, pic=True, dist_func='adj_cos', prefix='layer' + str(layer_num) + '_test2test_' + prefix, suffix='_txt2img_adjcos_' + suffix, save_results=False, save_path=save_path+'adjcos')
    print '3.2测试集图像检索测试及文本:'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2test_img2txt' + suffix, X_train=X_txt_test, label_train=y_test, X_test=X_img_test, label_test=y_test, pic=True, dist_func='adj_cos', prefix='layer' + str(layer_num) + '_test2test_' + prefix, suffix='_img2txt_adjcos_' + suffix, save_results=False, save_path=save_path+'adjcos')

    print '4.1测试集文本检索训练集图像'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2train_txt2img' + suffix, X_train=X_img_train, label_train=y_train, X_test=X_txt_test, label_test=y_test, pic=True, dist_func='adj_cos', prefix='layer' + str(layer_num) + '_test2train_' + prefix, suffix='_txt2img_adjcos_' + suffix, save_results=False, save_path=save_path+'adjcos')
    print '4.2测试集图像检索训练集文本'
    p_r(title=prefix + 'layer' + str(layer_num) + '_test2train_img2txt' + suffix, X_train=X_txt_train, label_train=y_train, X_test=X_img_test, label_test=y_test, pic=True, dist_func='adj_cos', prefix='layer' + str(layer_num) + '_test2train_' + prefix, suffix='_img2txt_adjcos_' + suffix, save_results=False, save_path=save_path+'adjcos')
    print 'evaluation finish...'
    
def finish_one_layer(X_img_train, X_txt_train, y_train, X_img_test, X_txt_test, y_test, h_units, epochs, lr=0.1, model_type='FullModal', alpha=0.8, layer_num='1', prefix='', suffix='', save_path=''):
    """预备+训练+测试完整的一层"""
    #1.构造数据集
    dsit_train, dsit_test = make_dataset(X_img_train=X_img_train, X_txt_train=X_txt_train, y_train=y_train, 
                                                        X_img_test=X_img_test, X_txt_test=X_txt_test, y_test=y_test)

    #2.训练单层模型
    monitoring_dataset = {'train': dsit_train, 'test': dsit_test}
	
    ae_model = MyMultimodalAutoEncoder(model_type=model_type, alpha=alpha, n_vis_img=X_img_train.shape[1], n_vis_txt=X_txt_train.shape[1], n_hid_img=h_units, n_hid_txt=h_units, dec_f_img=True, dec_f_txt=True)
    alg = SGD(learning_rate=lr, cost=None, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset, termination_criterion=EpochCounter(max_epochs=epochs))
    
    train = Train(dataset=dsit_train, model=ae_model, algorithm=alg, save_path='multi_ae_save_layer' + layer_num + '.pkl', save_freq=10)
    
    t0 = time.clock()
    train.main_loop()
    print 'training time for layer%s: %f' % (layer_num, time.clock() - t0)
    
    #3.计算经过训练后模型传播的设计矩阵
    X_img_propup_train, X_txt_propup_train, X_img_propup_test, X_txt_propup_test, X_propup_train, X_propup_test = propup_design_matrix(X_train=dsit_train.X, X_test=dsit_test.X, ae_model=ae_model)
    
    #4.测试训练后的模型分类性能
    print '!!!evaluate model on dataset+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    model_evaluate(X_img_train=X_img_propup_train, X_txt_train=X_txt_propup_train, y_train=y_train, X_img_test= X_img_propup_test, X_txt_test=X_txt_propup_test, y_test=y_test, layer_num=layer_num, prefix=prefix, suffix=suffix, save_path=save_path)
    
    return X_img_propup_train, X_txt_propup_train, X_img_propup_test, X_txt_propup_test

