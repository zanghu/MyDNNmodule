# -*- coding: utf-8 -*-
"""
Honglak Lee's papper: Sparse deep belief net model for visual area V2, NIPS 2007.
This code is mainly about sparse Gaussian-Binary RBM model.
Honglak Lee set sigma to 0.4 and 0.05 for the first and second layers
we used p = 0.02 and 0.05 for the first and second layers.
lambda = 1/p in each layer
"""
import time
import os
import copy

import numpy
import theano
#import theano.tensor as T
import scipy

from collections import OrderedDict

theano.config.compute_test_value = 'off'
#assert theano.config.floatX == 'float64'

def sigmoid(X):
    """
    sigmoid of X
    """
    return (1. + scipy.tanh(X/2))/2.
    
# Is that necessary to inherit Layer class??
class MyGaussianBinaryRBM(object):
    """Gaussian-Binary Restricted Boltzmann Machine (GB-RBM)  """
    def __init__(self, n_vis, n_hid, sigma=0.4, W=None, h_bias=None, v_bias=None, numpy_rng=None):
        """ """
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.sigma = sigma
        self.coeff = 1. / (self.sigma**2) #coefficient
        self.HL_p = 0.02

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(seed=19920130)
        self.numpy_rng = numpy_rng

        if W is None:
            W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hid + n_vis)),
                      high=4 * numpy.sqrt(6. / (n_hid + n_vis)),
                      size=(n_vis, n_hid)),
                      dtype=theano.config.floatX)
        else:
            assert isinstance(W, numpy.ndarray)
            assert W.ndim == 2
            
        if h_bias is None:
            h_bias = numpy.zeros(n_hid, dtype=theano.config.floatX)
        else:
            assert isinstance(h_bias, numpy.ndarray)
            assert h_bias.get_value().ndim == 1

        if v_bias is None:
            v_bias = numpy.zeros(n_vis, dtype=theano.config.floatX)
        else:
            assert isinstance(W, numpy.ndarray)
            assert v_bias.get_value().ndim == 1

        self.W = W
        self.h_bias = h_bias
        self.v_bias = v_bias

        self._params = [self.W, self.h_bias, self.v_bias]
        
    def train(self, data, batch_size, lr, epochs, k=1, momentum=0.0):
        
        self.data = data
        #self.k = k # cd-k
        self.lr = lr
        self.momentum = float(momentum)
        self.batch_size = batch_size
        # weight updates
        self.wu_vh = numpy.zeros((self.n_vis, self.n_hid))
        self.wu_v = numpy.zeros((self.n_vis))
        self.wu_h = numpy.zeros((self.n_hid))
        self.epochs = epochs
        self.batches = data.shape[0]/self.batch_size
        
        #err_list = self.cd_k_updates_check(k=1)
        err_list = self.cd_k_updates_HLSparse(k=1)

        return {"w": self.W, 
                "b": self.v_bias, 
                "b": self.h_bias,
                "err": err_list}
    
    def HL_sparse(self, v0, p=0.02):
        """function used for compute Honglak Lee's Sparse"""
        mean_matrix = self.propup(v0)
        part_j = p - mean_matrix.mean(axis=0)
        part_i1_matrix = mean_matrix * (1. - mean_matrix)
        #part_i2_vector = v0
        part_i = numpy.dot(v0.T, part_i1_matrix)
        part_orin = part_i * part_j #矩阵右乘一个行向量
        coeff = -2. / self.batch_size * self.coeff
        gW = coeff * part_orin #HL sparse项产生的梯度，不含lambda_
                
        part_j1 = part_j
        part_j2 = part_i1_matrix.mean(axis=0)
        coeff = -2. * self.coeff # * part_j1 * part_j2
        gc = coeff * part_j1 * part_j2
        
        return [1./p*gW, 1./p*gc]

    def HL_sparse_bias_only(self, v0, p=0.02):
        """function used for compute Honglak Lee's Sparse"""
        mean_matrix = self.propup(v0)       
        part_j1 = p - mean_matrix.mean(axis=0)
        part_j2 = (mean_matrix * (1. - mean_matrix)).mean(axis=0)
        coeff = -2. * self.coeff
        gc = coeff * part_j1 * part_j2
        
        return 1./p*gc
    
    def sparseness(self, v):
        
        h = self.propup(v)
        sparseness_vector = (numpy.sqrt(h.shape[1]) - numpy.sum(numpy.abs(h), axis=1)/numpy.sqrt(numpy.sum(h**2, axis=1))) / (numpy.sqrt(h.shape[1]) - 1.)
        sparseness = numpy.mean(sparseness_vector)
        return sparseness

    def cd_k_updates_HLSparse(self, k):
        """HL sparse penalty has been included in Cost function"""
        print 'recon_per_epoch'
        delta = self.lr / self.batch_size
        print "learning_rate: %f" % self.lr
        print "updates per epoch: %s | total updates: %s"%(self.batches, self.batches*self.epochs)
        err_list = []
        for epoch in xrange(self.epochs):
            print "[GB-RBM_NUMPY] Epoch", epoch
            err = []
            spa = []
            for batch_num in xrange(self.batches):
                v0 = self.data[batch_num*self.batch_size: (batch_num+1)*self.batch_size] # 本次迭代所用的batch对应的样本v1
                # cd-k approximation
                [h0_mean, h0_sample, vk_mean, vk_sample, hk_mean, hk_sample] = self.cd_k_alg(k=k, v0=v0)

                h0 = h0_mean 
                vk = vk_sample
                hk = hk_mean
                
                #Honglak Lee's Sparse
                gc_HL = self.HL_sparse_bias_only(v0=v0, p=self.HL_p)
                
                # compute updates
                self.wu_vh = self.wu_vh * self.momentum + self.coeff * (numpy.dot(v0.T, h0) - numpy.dot(vk.T, hk))# + self.batch_size*gW_HL
                self.wu_v = self.wu_v * self.momentum + self.coeff * (v0.sum(axis=0) - vk.sum(axis=0))
                self.wu_h = self.wu_h * self.momentum + self.coeff * (h0.sum(axis=0) - hk.sum(axis=0)) - self.batch_size*gc_HL
                # update 
                self.W += self.wu_vh * delta 
                self.v_bias += self.wu_v * delta
                self.h_bias += self.wu_h * delta
                # calculate reconstruction error
                err.append(((vk_mean-v0)**2).mean(axis=0).sum())
                spa.append(self.sparseness(v0))
            mean = numpy.mean(err)
            mean_spa = numpy.mean(spa)
            print "Mean squared error: " + str(mean)
            print "activation sparseness: " + str(mean_spa)
            err_list.append(float(mean))
        return err_list
    
    def cd_k_updates(self, k):
        """Honglak Lee' Sparse term has not been included in Cost function"""
        print 'recon_per_epoch'
        delta = self.lr / self.batch_size
        print "learning_rate: %f" % self.lr
        print "updates per epoch: %s | total updates: %s"%(self.batches, self.batches*self.epochs)
        err_list = []
        for epoch in xrange(self.epochs):
            print "[GB-RBM_NUMPY] Epoch", epoch
            err = []
            spa = []
            for batch_num in xrange(self.batches):
                v0 = self.data[batch_num*self.batch_size: (batch_num+1)*self.batch_size] # 本次迭代所用的batch对应的样本v1
                # cd-k approximation
                [h0_mean, h0_sample, vk_mean, vk_sample, hk_mean, hk_sample] = self.cd_k_alg(k=k, v0=v0)

                h0 = h0_mean 
                vk = vk_sample
                hk = hk_mean
                
                # compute updates
                self.wu_vh = self.wu_vh * self.momentum + self.coeff * (numpy.dot(v0.T, h0) - numpy.dot(vk.T, hk))
                self.wu_v = self.wu_v * self.momentum + self.coeff * (v0.sum(axis=0) - vk.sum(axis=0))
                self.wu_h = self.wu_h * self.momentum + self.coeff * (h0.sum(axis=0) - hk.sum(axis=0))
                # update 
                self.W += self.wu_vh * delta 
                self.v_bias += self.wu_v * delta
                self.h_bias += self.wu_h * delta
                # calculate reconstruction error
                err.append(((vk_mean-v0)**2).mean(axis=0).sum())
                spa.append(self.sparseness(v0))
            mean = numpy.mean(err)
            mean_spa = numpy.mean(spa)
            print "Mean squared error: " + str(mean)
            print "activation sparseness: " + str(mean_spa)
            err_list.append(float(mean))
        return err_list
    
    def cd_k_updates_check(self, k, HL_sparse=True): 
        """compute recon_error per epoch, with slower speed, only used for checking code correct"""
        print 'recon_per_epoch'
        delta = self.lr / self.batch_size
        print "learning_rate: %f" % self.lr
        print "updates per epoch: %s | total updates: %s"%(self.batches, self.batches*self.epochs)
        err_list = []
        for epoch in xrange(self.epochs):
            print "[GB-RBM_NUMPY] Epoch", epoch
            #err = []
            #spa = []
            for batch_num in xrange(self.batches):
                v0 = self.data[batch_num*self.batch_size: (batch_num+1)*self.batch_size] # 本次迭代所用的batch对应的样本v1
                # cd-k approximation
                [h0_mean, h0_sample, vk_mean, vk_sample, hk_mean, hk_sample] = self.cd_k_alg(k=k, v0=v0)

                h0 = h0_mean 
                vk = vk_sample
                hk = hk_mean
                
                if HL_sparse is True:
                #Honglak Lee's Sparse
                    gW_HL, gc_HL = self.HL_sparse(v0=v0, p=self.HL_p)
                else:
                    gW_HL = 0.
                    gc_HL = 0.
                
                # compute updates
                self.wu_vh = self.wu_vh * self.momentum + self.coeff * (numpy.dot(v0.T, h0) - numpy.dot(vk.T, hk)) #- self.batch_size*gW_HL
                self.wu_v = self.wu_v * self.momentum + self.coeff * (v0.sum(axis=0) - vk.sum(axis=0))
                self.wu_h = self.wu_h * self.momentum + self.coeff * (h0.sum(axis=0) - hk.sum(axis=0)) - self.batch_size*gc_HL
                # update 
                self.W += self.wu_vh * delta 
                self.v_bias += self.wu_v * delta
                self.h_bias += self.wu_h * delta
            # calculate reconstruction error
            err = []
            spa = []
            for batch_num in xrange(self.batches):
                v0 = self.data[batch_num*self.batch_size: (batch_num+1)*self.batch_size]
                _, _, vk_mean, vk_sample = self.gibbs_vhv(v0_sample=v0)
                err.append(((vk_mean-v0)**2).mean(axis=0).sum())
                spa.append(self.sparseness(v0))
            mean = numpy.mean(err)
            mean_spa = numpy.mean(spa)
            print "Mean squared error: " + str(mean)
            print "activation sparseness: " + str(mean_spa)
            err_list.append(float(mean))
        return err_list
    
        # cd-k算法的核心抽样部分，其实就是一次交替Gibbs抽样，该方法被cd_k_updates()和cd_k_updates_1()调用
    def cd_k_alg(self, k, v0, reconstruction=False): 
        #chain_start = h1
        v_list = []; v_mean_list = []
        h_list = []; h_mean_list = []
        h0_mean, h0 = self.sample_h_given_v(v0)
        h_list.append(h0); h_mean_list.append(h0_mean)
        for i in xrange(k):
            v_mean, v, h_mean, h = self.gibbs_hvh(h_list[-1])
            v_list.append(v); v_mean_list.append(v_mean)
            h_list.append(h); h_mean_list.append(h_mean)
        return h_mean_list[0], h_list[0], v_mean_list[-1], v_list[-1], h_mean_list[-1], h_list[-1]
    
    def get_params(self):
        
        return self._params

    def propup(self, v):
        """"""
        sigmoid_activation = self.coeff * (numpy.dot(v, self.W) + self.h_bias)
        return sigmoid(sigmoid_activation)

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        h0_mean = self.propup(v0_sample)
        #h0_sample = self.numpy_rng.binomial(n=1, p=h0_mean, dtype=theano.config.floatX)
        h0_rand = self.numpy_rng.rand(v0_sample.shape[0], self.n_hid)
        h0_sample = numpy.array(h0_rand < h0_mean, dtype=int)
        return [h0_mean, h0_sample]

    def propdown(self, h):
        
        Gaussian_mean = self.v_bias + numpy.dot(h, self.W.T)
        return Gaussian_mean

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.normal(loc=v1_mean, scale=self.sigma)
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



