import numpy
from dataset_from_design import DatasetFromDesign
import cPickle

def load_pylearn2_feret55_sampled(which_set):
    """"""
    assert which_set in ['train', 'test']
    if which_set == 'train':
        f = open('/home/zanghu/feret55_train_X_sampled.pkl', 'r')
    else:
        f = open('/home/zanghu/feret55_test_X_sampled.pkl', 'r')
    dsf = cPickle.load(f)
    f.close()
    return dsf

