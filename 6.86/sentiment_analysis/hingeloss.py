
import numpy as np

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    hg_loss = 0
    print (label*(np.dot(feature_vector.T,theta) + theta_0))
    if (label*(np.dot(feature_vector.T,theta) + theta_0) <= 1):
        hg_loss = 1 - label*(np.dot(feature_vector.T,theta) + theta_0)

    return (hg_loss)
    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    hg_loss = 0
    for i,feature_vector in enumerate(feature_matrix):
        label = labels[i]
        if (label*(np.dot(feature_vector.T,theta) + theta_0) <= 1):
            hg_loss = hg_loss + (1 - label*(np.dot(feature_vector.T,theta) + theta_0))

    return (hg_loss/feature_matrix.shape[0])
    raise NotImplementedError


# feature_vector = np.array([-0.23789775,-0.0394935,  -0.72518719, -0.891119,   -0.44038038, -0.48217244,
#  -0.52157629, -0.63730334, -0.27681579, -0.33903769])
# label = 1.0
# theta = [0.08383202, 0.95211259, 0.11326864, 0.43072054, 0.3282253,  0.92182662, 0.17523307, 0.9935178,  0.97708063, 0.60700495]
# theta_0 = 0.4593965835036775

feature_matrix = np.array([[0.14358771, 0.41213299, 0.59383735, 0.85316691, 0.57563801, 0.17039271, 0.67282569, 0.02622133, 0.37205599, 0.4289227 ],[-0.23789775,-0.0394935,  -0.72518719, -0.891119,   -0.44038038, -0.48217244,
 -0.52157629, -0.63730334, -0.27681579, -0.33903769]])
labels = [1,1]
theta = [-0.46950512, -0.79644072,  0.8692572,  -0.02037549,  0.10138906, -0.4372861, 0.66207093 ,-0.50855361 ,-0.15698099,  0.33726055]
theta_0= 0

print (hinge_loss_full(feature_matrix, labels, theta, theta_0))