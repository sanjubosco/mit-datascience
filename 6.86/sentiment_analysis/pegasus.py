import numpy as np 


import random
def get_order(rows):
    rows_index = list(range(rows))
    random.shuffle(rows_index)
    return rows_index

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    if (label*(np.dot(feature_vector.T,current_theta) + current_theta_0) <= 1):
        current_theta = (1-eta*L)*current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label
    else:
        current_theta = (1-eta*L)*current_theta
    print (current_theta)
    return (current_theta,current_theta_0)
    raise NotImplementedError


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Your code here

    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0
    updates = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            updates += 1
            eta = 1/np.sqrt(updates)
            current_theta,current_theta_0 = pegasos_single_step_update(feature_matrix[i],labels[i],L,eta,current_theta,current_theta_0)

    return (current_theta, current_theta_0)  
    raise NotImplementedError


feature_matrix = np.array([[0.1837462,0.29989789,-0.35889786,-0.30780561,-0.44230703,-0.03043835,0.21370063,0.33344998,-0.40850817,-0.13105809],
[0.08254096,0.06012654,0.19821234,0.40958367,0.07155838,-0.49830717,0.09098162,0.19062183,-0.27312663,0.39060785],
[-0.20112519,-0.00593087,0.05738862,0.16811148,-0.10466314,-0.21348009,0.45806193,-0.27659307,0.2901038,-0.29736505],
[-0.14703536,-0.45573697,-0.47563745,-0.08546162,-0.08562345,0.07636098,-0.42087389,-0.16322197,-0.02759763,0.0297091,],
[-0.18082261,0.28644149,-0.47549449,-0.3049562,0.13967768,0.34904474,0.20627692,0.28407868,0.21849356,-0.01642202]])
labels = np.array([-1,-1,-1,1,-1])
T = 10
L = 0.1456692551041303

current_theta, current_theta_0 = pegasos(feature_matrix, labels, T, L)
print (current_theta)