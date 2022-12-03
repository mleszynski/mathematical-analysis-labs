# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
Marcelo Leszynski
Math 347 Sec 003
02/23/21
"""
import numpy as np
import math
from scipy import linalg as la
from scipy import stats
from matplotlib import pyplot as plt


# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # points are randomly sampled and stored as an array of tuples #############
    points = np.random.uniform(-1, 1, (n,N))
    lengths = la.norm(points, axis=0)

    # count number of points within unit sphere ################################
    num_within = np.count_nonzero(lengths < 1)

    return (2**n) * (num_within/N)


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    # initialize list of points ################################################
    values = [f(x) for x in np.random.uniform(a,b,N)]
    return (b-a) * np.sum(values) / N

# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    # calculate the volume of the integrating area #############################
    lengths = [maxs[i] - mins[i] for i in range(len(maxs))]
    volume = np.prod(lengths)

    # sample and scale function sampling points ################################
    points = np.random.uniform(0,1,(N, len(maxs)))
    # iterate through points ###################################################
    for point in points:
        # iterate through coordinates in a point ###############################
        for i in range(len(point)):
            point[i] = (point[i] * (maxs[i]-mins[i])) + mins[i]

    # calculate values and return integral estimate ############################
    values = [f(point) for point in points]
    return volume * np.sum(values) / N


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    # define function and volume of integration ################################
    mins = [-3/2, 0 ,0, 0]
    maxs = [3/4, 1, 1/2, 1]
    f = lambda x: math.e**(-(x@x)/2)/(2*math.pi)**(len(x)/2)

    # calculate actual value of the integral of f ##############################
    means, cov = np.zeros(4), np.eye(4)
    accurate_val = stats.mvn.mvnun(mins, maxs, means, cov)[0]

    # estimate using estimations of size N #####################################
    N_domain = np.logspace(1,5,20, dtype='int')
    estimates = [mc_integrate(f, mins, maxs, N) for N in N_domain]
    rel_errors = [np.abs(accurate_val - estimate)/np.abs(accurate_val) for estimate in estimates]
    comparison = [1/np.sqrt(N) for N in N_domain]

    # plot the results #########################################################
    plt.loglog(N_domain, rel_errors, label='Relative Error')
    plt.loglog(N_domain, comparison, label='1/np.root(N)')
    plt.legend(loc='upper right')

    plt.show()
