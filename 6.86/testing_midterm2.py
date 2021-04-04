import numpy as np 
import itertools


# Y = np.array([[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1]])
# X = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]])
# expected_errors = np.array([1,9,10,5,9,11,0,3,1,1])

Y = np.array([[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1]])
X = np.array([[0,0],[2,0],[1,1],[0,2],[3,3],[4,1],[5,2],[1,4],[4,4],[5,5]])
expected_errors = np.array([1,65,11,31,72,30,0,21,4,15])
kernel_list = list()
sqrt_2 = np.sqrt(2)
for x in X:
    kernel_list.append([x[0]**2, sqrt_2*x[0]*x[1],x[1]**2])

X = np.array(kernel_list)

expected_errors = np.reshape(expected_errors,(10,1))
z = np.multiply(Y,expected_errors)
print (Y.T)
print (expected_errors.T)
print (z.T)

total = np.multiply(X,z)
print (X.T)
print (total)
print (np.sum(total,axis=0))
print (np.sum(z))

