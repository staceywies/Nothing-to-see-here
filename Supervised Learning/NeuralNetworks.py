import numpy as np

#Data
strdata=np.loadtxt('fertility.csv', delimiter=',', dtype=str)
input=strdata[:,0:-1]
input=input.astype('float')
output=strdata[:,-1:]
output[output== 'N'] = 0 #normal fertility is a 0
output[output== 'O'] = 1 #abnormal fertility is 1
output=output.astype('float')