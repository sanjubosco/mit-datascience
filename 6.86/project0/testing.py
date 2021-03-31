
import numpy as np 

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid_aprx(x):
    if (x >= 1):
        return 1
    elif ( x <= -1):
        return 0
    else:
        return (1/(1+np.exp(-x))) 

def tanh_aprx(x):
    if (x >= 1):
        return 1
    elif ( x <= -1):
        return -1
    else:
        return (np.tanh(x))


def lstm():
    X = np.array([0,0,1,1,1,0])
    H = list()
    h_prev = 0
    c_prev = 0
    for x in X:
        ft = sigmoid(-100)
        it = sigmoid((100*x) + 100)
        ot = sigmoid(100*x)
        ct = ft*c_prev + it*(np.tanh(-100*h_prev + 50*x))
        ht = ot*np.tanh(ct)
        H.append(ht)
        h_prev = ht
        c_prev = ct

    print (H)

def lstm_aprox():
    X = np.array([1,1,0,1,1])
    H = list()
    H_full = list()
    h_prev = 0
    c_prev = 0
    for x in X:
        ft = sigmoid_aprx(-100)
        it = sigmoid_aprx((100*x) + 100)
        ot = sigmoid_aprx(100*x)
        ct = ft*c_prev + it*(tanh_aprx(-100*h_prev + 50*x))
        H_full.append(ot*np.tanh(ct))
        ht = round(ot*tanh_aprx(ct))        
        H.append(ht)
        h_prev = ht
        c_prev = ct

    print (H)
    print (H_full)



def backpropagation():
    x = 3
    t = 1
    w1 = 0.01
    w2 = -5
    b = -1

    z1 = w1*x
    a1 = z1
    z2 = (w2*a1) + b
    y = sigmoid(z2)
    C = ((y-t)**2)/2
    z2_exp = np.exp(-z2)
    d_a2_z2 = z2_exp/((1+z2_exp)**2)
    d_c_w1 = x*w2*d_a2_z2*(y-t)
    print (d_c_w1)

backpropagation()


