import top

class Trainer(object):
    def __init__(self, sc, dataset):
        self.sc = sc
        self.dataset = dataset
        self.bsize = self.sc.model.batch_size
        self.esize = self.dataset.get_num_examples() / self.bsize
        
        cost = self.sc.cost() 
        self.opt = top.Optimizer(self.sc.model.get_params(), cost,
                input=self.sc.s, method='sgd', learning_rate=0.006)
    
    def run(self):
        i = 0
        for b in self.dataset.iterator('shuffled_sequential', self.bsize, self.esize):
            i+=1
            print 'Batch #:%d' % i
            self.sc.update_code(b)
            self.opt.run(1,b)

