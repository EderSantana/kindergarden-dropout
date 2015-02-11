"""
Most of this code was copied from breze library 
"""

import functools
import cPickle
import numpy as np
import scipy

from pylearn2.space import VectorSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.sandbox.rnn.space import SequenceDataSpace, SequenceSpace
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator
SIZE = 108 - 21 + 1
def padzeros(lst, front=True, return_mask=False):
    # TODO add docs for ``front``
    """Given a list of arrays, pad every array with up front  zeros until they
    reach unit length.
    Each element of `lst` can have a different first dimension, but has to be
    equal on the other dimensions.
    """
    n_items = len(lst)
    # Get the longest item.
    maxlength = max(len(i) for i in lst)
    restshape = list(lst[0].shape)[1:]
    item_shape = [maxlength] + restshape
    total_shape = [n_items] + item_shape

    data = scipy.zeros(total_shape, dtype=lst[0].dtype)
    if return_mask:
        mask = scipy.zeros(total_shape, dtype=lst[0].dtype)
    for i in range(n_items):
        # Iterate over indices because we work in place of the list.
        thislength = lst[i].shape[0]
        if front:
            data[i][-thislength:] = lst[i]
            if return_mask:
                mask[i][-thislength:] = 1
        else:
            data[i][:thislength] = lst[i]
            if return_mask:
                mask[i][:thislength] = 1

    if return_mask:
        return data, scipy.asarray(mask)
    return data

def interleave(lst):
    """Given a list of arrays, interleave the arrays in a way that the
    first dimension represents the first dimension of every array.
    This is useful for time series, where multiple time series should be
    processed in a single swipe."""
    arr = scipy.asarray(lst)
    return scipy.swapaxes(arr, 0, 1)

def split(X, maxlength):
    """Return a list of sequences where each sequence has a length of at most
    `maxlength`.
    Given a list of sequences `X`, the sequences are split accordingly."""
    new_X = []
    for seq in X:
        n_new_seqs, rest = divmod(seq.shape[0], maxlength)
        if rest:
            n_new_seqs += 1
        for i in range(n_new_seqs):
            new_X.append(seq[i * maxlength:(i + 1) * maxlength])
    return new_X

def masked(idxs):
    x = np.zeros(SIZE)
    x[(np.array(idxs) - 21).tolist()] = 1
    return x


def rolls_from_sequences(seqs):
    x = []
    for seq in seqs:
        x.append([])
        for item in seq:
            x[-1].append(masked(item))
        x[-1] = np.array(x[-1])
    return x


def load_data(handle):
    with open('%s.pickle' % handle) as fp:
        data = cPickle.load(fp)

    train, valid, test = data['train'], data['valid'], data['test']

    x = rolls_from_sequences(data['train'])
    vx = rolls_from_sequences(data['valid'])
    tx = rolls_from_sequences(data['test'])

    #print 'size of x', sum([i.size for i in x])
    #print 'size of vx', sum([i.size for i in vx])

    x = split(x, 100)
    vx = split(vx, 100)

    # Standardize.
    #scaler = StandardScaler()
    #x, (vx, tx) = static_transform([scaler], x, [vx, tx])

    X = interleave(padzeros(x, False))
    VX = interleave(padzeros(vx, False))

    #print 'size of X', X.size
    #print 'size of VX', VX.size

    Z, VZ, tz = X[1:], VX[1:], [i[1:] for i in tx]
    X, VX, tx = X[:-1], VX[:-1], [i[:-1] for i in tx]

    X, VX, Z, VZ = [i.astype('float32') for i in (X, VX, Z, VZ)]
    tx = [i.astype('float32') for i in tx]
    tz = [i.astype('float32') for i in tz]

    #print 'training shape', X.shape
    #print 'validation shape', VX.shape

    X = X.transpose(1,0,2)
    Z = Z.transpose(1,0,2)
    VX = VX.transpose(1,0,2)
    VZ = VZ.transpose(1,0,2)

    return X, Z, VX, VZ, tx, tz

class MID(DenseDesignMatrix):
    """
        TODO:
            Writeme

    """
    def __init__(self,which_set,source_file):
        self.which_set = which_set
        self.source_file = source_file
        X, Z, VX, VZ, tx, tz = load_data('/Users/eder/Copy/python/data/midi/JSBChorales')
        
        if self.which_set == 'train':
            x = X
            y = Z
        elif self.which_set == 'valid':
            x = VX
            y = VZ
        elif self.which_set == 'test':
            print 'This is probably broken!'
            x = tx
            y = tz
        else:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train", "valid", "test"].')
            
        self.time_length = x.shape[1]
        #x = x.reshape([-1, x.shape[-1]])
        #y = y.reshape([-1, y.shape[-1]])

        super(MID, self).__init__(X=x, y=y)
        
        self.X_space = SequenceSpace(VectorSpace(dim=x.shape[-1]))

    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):
        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    @functools.wraps(VectorSpacesDataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode=None):
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        # This should be fixed to allow iteration with default data_specs
        # i.e. add a mask automatically maybe?
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)
                
