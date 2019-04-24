# from TrainCNN import run
import TrainCNN as tc
import numpy as np

d = []

w1 = np.load('w1.npy')
w2 = np.load('w2.npy')
d.append(np.load('d0.npy'))

tc.run(tc.n[70][0], d[0], w1, w2)
tc.run(tc.o[70][0], d[0], w1, w2)
