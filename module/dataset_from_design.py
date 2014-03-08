# -*- coding: utf-8 -*-  
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
import numpy

class DatasetFromDesign(DenseDesignMatrix):
    """my module"""
    def __init__(self, design_matrix, labels=None, one_hot=False, classes=None, view_converter=None):
        
        #assert (which_set in ['train', 'test'])

        if one_hot is True and labels.ndim == 1:# 如果输入的labels本身不是向量(即labels.ndim==2)，则认为已经是one-hot的形式
            assert labels is not None
            assert classes is not None
            y_temp = numpy.zeros(shape=(labels.shape[0], classes))
            for i in xrange(labels.shape[0]):
                y_temp[labels[i]] = 1
            labels = y_temp
        super(DatasetFromDesign, self).__init__(X=design_matrix, y=labels)
        self.view_converter = view_converter
