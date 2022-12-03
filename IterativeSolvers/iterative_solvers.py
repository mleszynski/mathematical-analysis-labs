# iterative_solvers.py
"""Volume 1: Iterative Solvers.
Marcelo Leszynski
Math 347 Section 3
6 April 2021
"""

import numpy as np
from scipy import sparse
from scipy import linalg as la
from matplotlib import pyplot as plt


# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A


# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    # initialize variables #####################################################
    n = len(b)
    d_inv = 1/np.diag(A)
    x_k = np.zeros(n)
    abs_error = []

    # iterate through to calculate error #######################################
    for iter in range(maxiter):
        x_k1 = x_k + (d_inv * (b - A@x_k))
        if la.norm(x_k1-x_k, np.inf) < tol:
            break

        abs_error.append(la.norm(A@x_k - b, np.inf))
        x_k = x_k1

    # problem 2 ################################################################
    if plot:
        plt.plot(range(iter), abs_error)
        plt.yscale("log")
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()

    return x_k1


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # initialize variables #####################################################
    n = len(b)
    d_inv = 1/np.diag(A)
    x_1 = np.zeros(n)
    abs_error = []

    # iterate through to calculate error #######################################
    for iter in range(maxiter):
        x_0 =  np.copy(x_1)
        for k in range(n):
            x_1[k] = x_0[k] + d_inv[k]*(b[k] - A[k].T@x_1)

        if la.norm(x_1 - x_0, np.inf) < tol:
            break

        abs_error.append(la.norm(A@x_0 - b, np.inf))

    # plot #####################################################################
    if plot:
        plt.plot(range(iter), abs_error)
        plt.yscale("log")
        plt.title("Convergence of Gauss Seidel Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()

    return x_1


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # initialize variables #####################################################
    n = len(b)
    d_inv = 1/np.diag(sparse.csr_matrix.toarray(A))
    x_1 = np.zeros(n)

    # iterate until tolerance is reached #######################################
    for iter in range(maxiter):
        x_0 =  np.copy(x_1)
        for k in range(n):
            row_start = A.indptr[k]
            row_end = A.indptr[k+1]
            Aix = A.data[row_start:row_end] @ x_1[A.indices[row_start:row_end]]
            x_1[k] = x_0[k] + d_inv[k]*(b[k] - Aix)

        if la.norm(x_1-x_0, np.inf) < tol:
            break

    return x_1


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    # initialize variables #####################################################
    n = len(b)
    d_inv = omega/np.diag(sparse.csr_matrix.toarray(A))
    x_1 = np.zeros(n)

    # iterate until tolerance is reached #######################################
    for iter in range(maxiter):
        x_0 =  np.copy(x_1)
        for k in range(n):
            row_start = A.indptr[k]
            row_end = A.indptr[k+1]
            Aix = A.data[row_start:row_end] @ x_1[A.indices[row_start:row_end]]
            x_1[k] = x_0[k] + d_inv[k]*(b[k] - Aix)

        if la.norm(x_1-x_0, np.inf) < tol:
            return x_1, True, iter+1

    return x_1, False, iter+1


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    # initialize plate values ##################################################
    B = sparse.diags([1,-4,1], [-1,0,1], shape=(n,n))
    A = sparse.block_diag([B]*n) + sparse.diags([1,1],[-n,n], shape=(n**2,n**2))
    b = np.array([-100 if i==0 or i==n-1 else 0 for i in range(n)])
    u, converged, num_iter = sor(A, np.tile(b,n), omega, tol, maxiter)

    # plot plate ###############################################################
    if plot:
        plt.pcolormesh(np.reshape(u,(n,n)), cmap='coolwarm')
        plt.show()

    return u, converged, num_iter

# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    # initialize variables #####################################################
    num_iters = []
    omegas = [i/100 for i in range(100, 200, 5)]

    # calculate values #########################################################
    for omega in omegas:
        num_iters.append(hot_plate(20, omega, 1e-2, 1000)[2])

    # plot values ##############################################################
    plt.plot(omegas, num_iters)
    plt.ylabel("Number of Iterations")
    plt.xlabel("Relaxation Factor")
    plt.show()

    return omegas[np.argmin(num_iters)]