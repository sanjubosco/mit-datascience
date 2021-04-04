import numpy as np 
import itertools
expected_errors = np.array([1,9,10,5,9,11,0,3,1,1])
roll = 12

# Y = np.array([[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1]])
# X = np.array([[0,0],[2,0],[3,0],[0,2],[2,2],[5,1],[5,2],[2,4],[4,4],[5,5]])

Y = np.array([[-1], [-1], [-1], [-1], [-1], [1], [1], [1], [1], [1]])
X = np.array([[0,0],[2,0],[1,1],[0,2],[3,3],[4,1],[5,2],[1,4],[4,4],[5,5]])
kernel_list = list()
sqrt_2 = np.sqrt(2)
for x in X:
    kernel_list.append([x[0]**2, sqrt_2*x[0]*x[1],x[1]**2])

X = np.array(kernel_list)

# Y = np.roll(Y,4)
# X = np.roll(X,4)
X_ALL = np.append(X,Y, axis = 1)
# X_delta = X_new[:3]
# XX = itertools.permutations(X_ALL)

W = np.array([21,-22.627417, 22, -110])
converged = True
for i,x_ in enumerate(X_ALL):
    x = x_.copy()
    y = x[-1]
    x[-1] = 1
    if (y*(np.dot(x,W.T)) <= 0):
        converged = False
        print ("sanju")
        break

if (converged):
    print ("Success")

exit()

for r in range(roll):
    X_new = np.roll(X_ALL,r,axis = 0)
    converged = False
    count = 0
    errors = np.zeros(X.shape[0])
    W = np.array([0,0,0,0]) 

    while (not(converged) and count < 10000):
        converged = True
        for i,x_ in enumerate(X_new):
            x = x_.copy()
            y = x[-1]
            x[-1] = 1

            if (y*(np.dot(x,W.T)) <= 0):
                W = W + y*x
                # W0 += Y[i]
                errors[i] += 1
                converged = False

        # for i,x_ in enumerate(X_new):
        #     x = x_.copy()
        #     y = x[-1]
        #     x[-1] = 1
        #     if (y*(np.dot(x,W.T)) <= 0):
        #         converged = False
        #         break

        if converged == True:
            break

        count += 1

    #expected_errors = [ 2., 12., 12.,  2.,  3., 14. , 0. , 0.,  0. , 0.]
    #if (np.array_equal(errors,expected_errors)):
    if converged == True:
        # print ("Found Match.")
        # print (converged, count)
        print (W)
        # #print (W0)
        # # print (X.T)
        # # print (errors)
        # print (X_new.T)
        # # print (Y)
        # print (errors)
        # print ("All errrors = ", np.sum(errors))
