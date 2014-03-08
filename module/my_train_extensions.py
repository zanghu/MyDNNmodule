import numpy
import theano
import theano.tensor as T
from pylearn2.train_extensions import TrainExtension

class Visualizer(TrainExtension):
    
    def on_monitor(self, model, dataset, algorithms):
    
        if not hasattr(model, 'get_visual'):
            raise Exception('')
        model.get_visual()

        return None
