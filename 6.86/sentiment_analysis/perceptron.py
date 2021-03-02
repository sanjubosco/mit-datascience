import numpy as np 

def test_perceptron():
    Xt = np.array([[np.cos(np.pi),0,0],[0,np.cos(2*np.pi),0],[0,0,np.cos(3*np.pi)]])
    Yt = [1,1,1]
    theta = [0,0,0]
    theta_list = list()

    print ("Initial values = X = %s, Y = %s, theta = %s" %(str(Xt), str(Yt), theta))

    for train in (range(10)):
        solution_found = True
        for i,x in enumerate(Xt):
            if (Yt[i]*(np.dot(x.T,theta)) <= 0):
                theta = theta + Yt[i]*x
                theta_list.append(theta)

        for i,x in enumerate(Xt):
            if (Yt[i]*(np.dot(x.T,theta)) <= 0):
                solution_found = False
                break

        if (solution_found):
            print ("Solution Found - Round = %d, theta = %s, theta_list = %s" %(train,str(theta), str(theta_list)))
            print ("theta_list = %s" %(str(theta_list)))
            break



def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here

    if (label*(np.dot(feature_vector.T,current_theta) + current_theta_0) <= 0):
        current_theta = current_theta + label*feature_vector
        current_theta_0 = current_theta_0 + label

    return (current_theta,current_theta_0)

    raise NotImplementedError


# feature_vector = np.array([[np.cos(np.pi),0,0],[0,np.cos(2*np.pi),0],[0,0,np.cos(3*np.pi)]])
# label = [1,1,1]

import random
def get_order(rows):
    rows_index = list(range(rows))
    random.shuffle(rows_index)
    return rows_index

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            current_theta, current_theta_0 = perceptron_single_step_update(
            feature_matrix[i],
            labels[i],
            current_theta,
            current_theta_0)

    return (current_theta, current_theta_0)            
    raise NotImplementedError

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0
    avg_theta = np.zeros(feature_matrix.shape[1])

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            current_theta, current_theta_0 = perceptron_single_step_update(
            feature_matrix[i],
            labels[i],
            current_theta,
            current_theta_0)
            avg_theta = avg_theta + current_theta
            print (current_theta,avg_theta)

    avg_theta = avg_theta/(T*feature_matrix.shape[0])
    
    return (avg_theta, current_theta_0)
    raise NotImplementedError


Xt = np.array([[np.cos(np.pi),0,0],[0,np.cos(2*np.pi),0],[0,0,np.cos(3*np.pi)]])
Yt = [1,1,1]
T= 10

current_theta,current_theta_0 = average_perceptron(Xt, Yt, T)
print (current_theta,current_theta_0)