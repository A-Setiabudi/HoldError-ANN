import sklearn as sk
from sklearn import preprocessing
import neurolab as nl
import pandas as pd

df = pd.read_excel('training.xlsx')
input = df.as_matrix()
# input_norm = sk.preprocessing.normalize(input)

df = pd.read_excel('target.xlsx')
target = df.as_matrix()
# target_norm = sk.preprocessing.normalize(target)


#net = nl.load('hold_error.net')
# input(Wp, Wn, Vdd, Vin, Ch)
net = nl.net.newff([[0, 1], [0, 1]], [20, 20, 20, 20, 20, 1])

net.trainf = nl.train.train_bfgs
# net.trainf = nl.train.train_gdx

error = net.train(input, target, epochs=5000, show=1, goal=1e-7)

net.save('hold_error1.net')

out = net.sim([[2e-6, 3.6e-5]])
print(out)
