# -*- coding: utf-8 -*-  
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
import numpy

class SiftMnist(DenseDesignMatrix):
    def __init__(self, which_set='train', design_matrix=None, labels=None, regularize=False, start=0, stop=None, one_hot=False):
        
        assert (which_set in ['train', 'valid', 'test'])
        if which_set == 'train':
            assert stop <= 11269
            if stop == None:
                stop = 11269
            f = open('/home/zanghu/sm_train.npy')
            X = numpy.load(f) #derive design-matrix
            f.close()
    
            f = open('/home/zanghu/sm_train_label.npy')
            y = numpy.load(f)
            f.close()
        if which_set == 'valid':
            assert stop <= 11269
            if stop == None:
                stop = 11269
            f = open('/home/zanghu/sm_valid.npy')
            X = numpy.load(f) #derive design-matrix
            f.close()
    
            f = open('/home/zanghu/sm_valid_label.npy')
            y = numpy.load(f)
            f.close()
        if which_set == 'test':
            assert stop <= 11269
            if stop == None:
                stop = 11269
            f = open('/home/zanghu/sm_test.npy')
            X = numpy.load(f) #derive design-matrix
            f.close()
    
            f = open('/home/zanghu/sm_test_label.npy')
            y = numpy.load(f)
            f.close()
        assert start <= stop
        if one_hot == True:
            Y = numpy.zeros(shape=(y.shape[0], 20), dtype='float32') 
            for i in xrange(y.shape[0]):
                Y[i, y[i]-1] = 1.
            y = Y
        
        if regularize:
            X_min = X.min(axis=1)
            X_max = X.max(axis=1)
            den = X_max - X_min
            X = (X - X_min[:, numpy.newaxis]) / den[:, numpy.newaxis]    
        if design_matrix is None:
            super(SiftMnist, self).__init__(X=X[start: stop,:], y=y[start: stop])
        else:
            super(SiftMnist, self).__init__(X=design_matrix, y=labels)
        self.view_converter = None
