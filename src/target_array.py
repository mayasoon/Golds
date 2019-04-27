import numpy as np

onodes = 10
targets = np.zeros(onodes) + 0.1
ss = np.transpose(np.array([1, 2, 3], ndmin=2))
print(ss.T)