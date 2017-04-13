#import xlrd
import neurolab as nl
import pandas as pd

df = pd.read_excel('training.xlsx')
input = df.as_matrix()

df = pd.read_excel('target.xlsx')
target = df.as_matrix()

# book = xlrd.open_workbook('training.xlsx')
# sheet = book.sheet_by_name('Sheet1')
# input = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]

# book = xlrd.open_workbook('target.xlsx')
# sheet = book.sheet_by_name('Sheet1')
# target = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]

print(input)
print(target)


# net = nl.load('sum.net')
# input(Wp, Wn, Vdd, Vin, Ch)
net = nl.net.newff([[0, 1e-4], [0, 1e-4], [0, 3], [0, 3] ,[0, 1e-9]],[50, 20, 20, 1])

net.trainf = nl.train.train_bfgs
#net.trainf = nl.train.train_gdx

error = net.train(input, target, epochs=5000, show=1, goal=0)

print(error)

net.save('hold_error.net')

out = net.sim([[1e-5, 1e-5, 3, 1.5, 1e-12], [2e-5, 2e-5, 3, 1.5, 1e-12], [5.5e-5, 4.9e-5, 3, 1.5, 1e-12]])

print(out)
