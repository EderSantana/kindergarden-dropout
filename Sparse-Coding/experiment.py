from sc import SC
from trainer import Trainer
from pylearn2.models.mlp import MLP, Linear
from pylearn2.datasets.mnist import MNIST

print 'Loading dataset'
dataset = MNIST(
        which_set = 'train',
        center = True
                )

hidden_size = 100
input_size = dataset.X_space.dim

print 'Creating model'
model = MLP(
            batch_size = 100,
            nvis = hidden_size,
            layers = [
                Linear(dim=input_size, layer_name='l1', irange=.05)
                ]
            )

print 'Creating Sparse Coder'
sc = SC(model, hidden_size, dataset.X_space)

print 'Training...'
trainer = Trainer(sc, dataset)

trainer.run()
