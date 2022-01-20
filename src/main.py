import numpy as np
from time import time
from train import train
import tensorflow as tf
from batch_test import dataloader,args

np.random.seed(555)
tf.set_random_seed(555) # since we have tf.random in the model


show_loss = False
show_time = False
show_topk = True

t = time()


print(args)
path = "../data/" +args.dataset
train(args, dataloader, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))
