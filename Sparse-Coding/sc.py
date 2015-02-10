import top
import theano
import numpy as np 

from theano import tensor as T
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace, Conv2DSpace
class SC(object):
    def __init__(self, model, hidden_size, input_space, steps=100):
        self.model = model
        
        bsize = self.model.batch_size
        if isinstance(input_space, VectorSpace):
            size = (bsize, hidden_size)
        elif isinstace(input_space, Conv2DSpace):
            #TODO this is wrong. Need to redefine image shape, 
            # it should be input shape - filter shape + 1.
            # also, num_channels should be defined by "hidden_size"
            size = (bsize, input_space.num_channels,
                    input_space.shape[0], 
                    input_space.shape[1], 
                    )
        else:
            raise NotImplemented('`input_space` not supported.')

        z = np.random.laplace(0,1,size=size)
        self.z = sharedX(z)
        self.s = input_space.make_theano_batch()
        self.steps = steps
        
        cost = self.cost()
        self.opt = top.Optimizer(self.z, cost, input=self.s, method='rmsprop',
                learning_rate=.0001, momentum=.9)

        self.compile()

    def cost(self):
        _,error = self.reconstruct(self.s)
        l1 = T.sqrt(self.z**2 + 1e-6).sum()
        rnorm = 0.
        for m in self.model.get_params():
            rnorm += T.sqr(m).sum()
        return error + .1 * l1 + .5 * rnorm

    def reconstruct(self, state):
        y = self.model.fprop(self.z)
        error = (state-y)**2
        return y, error.sum()
    
    def update_code(self, state):
        self.opt.run(self.steps, state)

    def compile(self):
        cost = self.cost()
        self.fcost = theano.function([self.s], cost, allow_input_downcast=True)
        y,_ = self.reconstruct(self.s)
        self.freconstruct = theano.function([], y)


