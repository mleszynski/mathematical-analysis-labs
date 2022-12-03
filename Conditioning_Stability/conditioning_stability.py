# condition_stability.py
"""Volume 1: Conditioning and Stability.
Marcelo Leszynski
Math 347 sec 003
02/20/21
"""

import numpy as np
import sympy as sy
import math
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    # calculate singular values ################################################
    s_vals = la.svdvals(A)

    # check to see if condition number is infinity #############################
    if s_vals[-1] == 0:
        return np.inf

    # calculate and return condition number ####################################
    return s_vals[0] / s_vals[-1]


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    w_roots = np.sort(np.arange(1, 21))
    
    new_roots = []
    abs_cond = []
    rel_cond = []

    # perform the experiment 100 times #########################################
    for i in range(100):
        perturbations = []

        # perturb the polynomial coefficients ##################################
        new_coeffs = w_coeffs.copy()
        for i in range(len(new_coeffs)):
            r_i = np.random.normal(1, 1e-10)
            perturbations.append(r_i)
        new_coeffs *= perturbations

        # calculate the new roots ##############################################        
        new_roots = np.sort(np.roots(np.poly1d(new_coeffs)))

        # plot the new roots ###################################################
        plt.plot(np.real(new_roots), np.imag(new_roots), 'k,')

        # calculate absolute condition number ##################################
        k_hat = la.norm(new_roots-w_roots, np.inf)/la.norm(np.array(perturbations), np.inf)
        abs_cond.append(k_hat)

        # calculate relative condition number ##################################
        k = la.norm(w_coeffs, np.inf)/la.norm(w_roots, np.inf)*k_hat
        rel_cond.append(k)

    # plot image ###############################################################
    plot_range = [0]*20
    #plot the last roots again to show up on the legend
    plt.plot(np.real(new_roots), np.imag(new_roots), 'k,', label='perturbed')  
    plt.plot(w_roots, plot_range,'bo', label='original')
    plt.legend(loc='upper left')
    plt.xlabel('real')
    plt.ylabel('imaginary')
    plt.show()

    # calculate and return condition numbers ###################################
    return np.sum(abs_cond) / len(abs_cond), np.sum(rel_cond) / len(rel_cond)


# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    # calculate and sort eigenvalues of A ######################################
    A_vals = la.eigvals(A)

    # create perturbating matrix ###############################################
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    # calculate and sort eigenvalues of new_A ##################################
    new_A = A + H
    new_A_vals = la.eigvals(new_A)

    # calculate and return condition numbers ###################################
    k_hat = la.norm(A_vals - new_A_vals, ord=2) / la.norm(H, ord=2)
    k = la.norm(A, ord=2) / la.norm(A_vals, ord=2) * k_hat

    return k_hat, k

# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    # initialize array for storing condition values ############################
    x_min, x_max, y_min, y_max = domain
    x_domain = np.linspace(x_min, x_max, res)
    y_domain = np.linspace(y_min, y_max, res)

    # compute x values, y values, and cond numbers #############################
    cond_values = np.array([[eig_cond(np.array([[1, x],[y,1]]))[1] for y in y_domain] for x in x_domain])

    # plot the colormesh map ###################################################
    plt.pcolormesh(cond_values, cmap='gray_r')
    plt.colorbar()
    plt.plot()
    plt.show()

# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    # load up necessary data ###################################################
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)
    
    # compute x using normal method ############################################
    normal_x = la.inv((A.T)@A)@(A.T)@yk
    
    # compute x using qr method ################################################
    Q, R = la.qr(A, mode='economic')
    qr_x = la.solve_triangular(R, (Q.T)@yk)

    # plot polynomials #########################################################
    domain = np.linspace(0,1,100)
    plt.plot(domain, np.polyval(normal_x, domain), color='blue', label='Normal Equations', marker='x')
    plt.plot(domain, np.polyval(qr_x, domain), color='orange', label='QR Solver')
    plt.plot(xk,yk, 'ko', label='Raw', markersize=0.5)
    plt.axis([0,1,0,25])
    plt.legend(loc='upper left')
    plt.show()

    # return forward errors ####################################################
    return la.norm((A@normal_x) - yk, ord=2), la.norm((A@qr_x) - yk, ord=2)

# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    # setup variables and sympy labels #########################################
    x = sy.Symbol('x')
    int_vals = []
    fact_vals = []
    domain = [n*5 for n in range(1,11)]

    # iterate through n values #################################################
    for n in domain:
        # integrate using sympy and store float values #########################
        int_vals.append(sy.integrate(x**int(n) * math.e**(x-1), (x,0,1)))

        # compute values using factorial method ################################
        fact_vals.append((-1)**n * (sy.subfactorial(int(n))-sy.factorial(int(n))/math.e))

    # compute forward errors ###################################################
    f_errors = [abs(int_vals[i] - fact_vals[i]) / abs(int_vals[i]) for i in range(len(int_vals))]

    # plot errors ##############################################################
    plt.plot(domain, f_errors, 'b-')
    plt.xlabel('N Values')
    plt.ylabel('Forward Error')
    plt.yscale('log')
    plt.show()