# differentiation.py
"""Volume 1: Differentiation.
Marcelo Leszynski
Math 347 Section 3
03/05/21
"""

import time
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from autograd import numpy as anp
from autograd import grad
from autograd import elementwise_grad


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    # initialize variables and function ########################################
    x = sy.symbols('x')
    f = (sy.sin(x)+1)**sy.sin(sy.cos(x))

    # calculate the derivative #################################################
    fprime = sy.diff(f, x)
    return sy.lambdify(x, fprime, "numpy")


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    # return the forward 1 coefficient #########################################
    return (f(x+h) - f(x)) / h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    # return the forward 2 coefficient #########################################
    return (-3*f(x) + 4*f(x+h) - f(x+2*h)) / (2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    # return the backward 1 coefficient ########################################
    return (f(x) - f(x-h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    # return the backward 2 coefficient ########################################
    return (3*f(x) - 4*f(x-h) + f(x-2*h)) / (2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    # return the centered 2 coefficient ########################################
    return (f(x+h) - f(x-h)) / (2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    # return the centered 4 coefficient ########################################
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h)) / (12*h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    # return first derivative from problem 1 ###################################
    f = lambda x: (np.sin(x)+1)**np.sin(np.cos(x))
    fprime = prob1()
    exact_value = fprime(x0)

    # calculate error bounds ###################################################
    error_f1 = []
    error_f2 = []
    error_b1 = []
    error_b2 = []
    error_c2 = []
    error_c4 = []

    h_log = np.logspace(-8,0,9)

    for h in h_log:
        error_f1.append(np.abs(exact_value - fdq1(f,x0,h)))
        error_f2.append(np.abs(exact_value - fdq2(f,x0,h)))
        error_b1.append(np.abs(exact_value - bdq1(f,x0,h)))
        error_b2.append(np.abs(exact_value - bdq2(f,x0,h)))
        error_c2.append(np.abs(exact_value - cdq2(f,x0,h)))
        error_c4.append(np.abs(exact_value - cdq4(f,x0,h)))

    # plot results #############################################################
    plt.loglog(h_log, error_f1, label="Order 1 Forward", marker="o")
    plt.loglog(h_log, error_f2, label="Order 2 Forward", marker="o")
    plt.loglog(h_log, error_b1, label="Order 1 Backward", marker="o")
    plt.loglog(h_log, error_b2, label="Order 2 Backward", marker="o")
    plt.loglog(h_log, error_c2, label="Order 2 Centered", marker="o")
    plt.loglog(h_log, error_c4, label="Order 4 Centered", marker="o")
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """

    # load and initialize data #################################################
    data = np.load("plane.npy")
    num_secs = len(data)

    a_array = [data[i,1] for i in range(num_secs)]
    b_array = [data[i,2] for i in range(num_secs)]
    speed_array = []

    # convert units to radians #################################################
    alpha_array = np.array(a_array) * np.pi / 180 
    beta_array = np.array(b_array) * np.pi / 180

    # create x and y equations #################################################
    a, B = sy.symbols('alpha, Beta')
    exp_X = (500*sy.tan(B)) / (sy.tan(B) - sy.tan(a))
    exp_Y = (500*sy.tan(B)*sy.tan(a)) / (sy.tan(B) - sy.tan(a))
    x = sy.lambdify((a,B), exp_X, "numpy")
    y = sy.lambdify((a,B), exp_Y, "numpy")

    # create the speeds array ##################################################
    for t in range(num_secs):
        # forward direction ####################################################
        if t == 0: 
            x_prime = x(alpha_array[t+1],beta_array[t+1]) - x(alpha_array[t],beta_array[t])
            y_prime = y(alpha_array[t+1],beta_array[t+1]) - y(alpha_array[t],beta_array[t])

        # backward direction ###################################################
        elif t == 7: 
            x_prime = x(alpha_array[t],beta_array[t]) - x(alpha_array[t-1],beta_array[t-1])
            y_prime = y(alpha_array[t],beta_array[t]) - y(alpha_array[t-1],beta_array[t-1])

        # centered #############################################################
        else: 
            x_prime = (x(alpha_array[t+1],beta_array[t+1]) - x(alpha_array[t-1],beta_array[t-1])) / 2
            y_prime = (y(alpha_array[t+1],beta_array[t+1]) - y(alpha_array[t-1],beta_array[t-1])) / 2

        speed_array.append(np.sqrt(x_prime**2 + y_prime**2))

    return speed_array


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    # get matrix dimensions ####################################################
    n = len(x)
    m = len(f(x))
    identity = np.eye(n)

    # calculate and return jacobian matrix #####################################
    jacobian = np.ones((m,n))
    for j in range(n):
        jacobian[:,j] = (f(x + h*identity[:,j]) - f(x - h*identity[:,j])) / (2*h)

    return jacobian


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    # handle base cases ########################################################
    if n == 0:
        return 1
    elif n == 1:
        return x
    
    # recursively calculate and return chebyshev polynomial ####################
    else:
        return 2*x * cheb_poly(x,n-1) - cheb_poly(x,n-2)


def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    # set up domain and derivatives ############################################
    domain = np.linspace(-1,1)
    prime = elementwise_grad(cheb_poly)

    # plot results #############################################################
    plt.plot(domain, anp.zeros_like(domain), label="n=0")
    for n in range(1,5):
        plt.plot(domain, prime(domain,n), label= "n=" + str(n))

    plt.legend()
    plt.show()


# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    # define initial functions #################################################
    x = sy.symbols('x')
    f = sy.lambdify(x, (sy.sin(x)+1)**sy.sin(sy.cos(x)), "numpy")
    g = lambda y: (anp.sin(y)+1)**anp.sin(anp.cos(y))

    # initialize times lists ###################################################
    exact_times = []
    exact_errors = np.full(shape=N, fill_value=1e-18)
    centered_times = []
    centered_errors = []
    autograd_times = []
    autograd_errors = []

    for i in range(N):
        # time exact times #####################################################
        x0 = np.random.randint(N)+ 1
        start = time.time()
        f_prime = prob1()
        f_x0 = f_prime(x0)
        exact_times.append(time.time()-start)
        # time centered times ##################################################
        start = time.time()
        c_x0 = cdq4(f,x0)
        centered_times.append(time.time()-start)
        centered_errors.append(np.abs(f_x0-c_x0))
        # time autograd times ##################################################
        start = time.time()
        aprime = grad(g)
        a_x0 = aprime(float(x0))
        autograd_times.append(time.time()-start)
        autograd_errors.append(np.abs(f_x0-a_x0))

    # plot results #############################################################
    plt.loglog(exact_times, exact_errors, linestyle="None", color="blue", marker="o", label="SymPy")
    plt.loglog(centered_times,centered_errors, linestyle="None", color="orange", marker="o", label="Difference Quotients")
    plt.loglog(autograd_times,autograd_errors, linestyle="None", color="green", marker="o", label="Autograd")
    plt.legend()
    plt.show()
