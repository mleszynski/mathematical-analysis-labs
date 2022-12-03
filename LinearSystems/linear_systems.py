# linear_systems.py
"""Volume 1: Linear Systems.
Marcelo Leszynski
Math 345 sec 001
10/13/20
"""
import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
from matplotlib import pyplot as plt
import time

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    # initialize num_rows and num_cols #########################################
    num_rows, num_cols = A.shape

    # use for-loop to compute REF(A) ###########################################
    for j in range(num_cols - 1):  # index across columns
        for i in range(j+1, num_rows):  # index across rows
            A[i,j:] -= (A[i,j] / A[j,j]) * A[j,j:]  # subtract the appropriate number from the row
    return A


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    # initialize variables #####################################################
    num_rows, num_cols = A.shape
    U = np.copy(A)
    L = np.identity(num_rows)

    # use for-loop to compute LU decomposition #################################
    for j in range(num_cols):
        for i in range(j+1, num_rows):
            L[i,j] = U[i,j] / U[j,j]
            U[i,j:] = U[i,j:] - L[i,j] * U[j,j:]
    return L, U

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    # initialize variables #####################################################
    L,U = lu(A)  # calculate the LU decomposition
    x = np.array([0]*len(b), dtype = np.float)
    y = np.array([0]*len(b), dtype = np.float)

    # use for-loop to calculate vector y #######################################
    for i in range(len(y)):
        y[i] = b[i] - np.dot(y, L[i])

    # use for-loop to calculate solution vector x ##############################
    for i in range(0,len(x)):
        # calculate the sum in 2.2. Negative indices are used as 
        # part of the "reverse" recursive relation of the equation given
        sum = np.dot(U[-(i+1),-i:], x[-i:])  
        x[-(i+1)] = 1 / U[-(i+1),-(i+1)] * (y[-(i+1)] - sum)
    return x


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    # initialize variables and empty times lists ###############################
    domain = np.arange(1,400)
    inv_times = []
    solve_times = []
    lu_factor_times = []
    lusolve_times = []

    # use for-loop to compute times for size n #################################
    for n in domain:
        # create random matrix A and vector b ##################################
        A = np.random.random((n,n))
        b = np.random.random(n)

        # time solving Ax = b using inv(A) #####################################
        start_time = time.time()
        inverse = la.inv(A)
        A @ b
        inv_times.append(time.time() - start_time)

        # time solving Ax = b using la.solve() #################################
        start_time = time.time()
        la.solve(A,b)
        solve_times.append(time.time() - start_time)

        # time solving Ax = b using LU decomposition and la.lu_solve() #########
        start_time = time.time()
        L, P = la.lu_factor(A)
        la.lu_solve((L,P),b)
        lu_factor_times.append(time.time() - start_time)

        # solve Ax = b using LU decomposition but only timing la.lu_solve() ####
        start_time = time.time()
        la.lu_solve((L,P),b)
        lusolve_times.append(time.time() - start_time)
    
    # plot results of timing different methods of solving Ax = b ###############
    plt.loglog(domain, inv_times, '.-', color = 'blue', linewidth = 1, markersize = 3, basex=2, basey=2, label = 'Matrix Inverse') 
    plt.loglog(domain, solve_times, '.-', color = 'orange', linewidth = 1, markersize = 3, basex=2, basey=2, label = 'la.solve')
    plt.loglog(domain, lu_factor_times, '.-', color = 'green', linewidth = 1, markersize = 3, basex=2, basey=2, label = 'LU and lu_solve') 
    plt.loglog(domain, lusolve_times, '.-', color = 'purple', linewidth = 1, markersize = 3, basex=2, basey=2, label = 'lu_solve only')
    plt.legend(loc="upper left")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("time", fontsize=14)
    plt.show()



# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    # construct the matrix B, then construct composite matrix A ################
    B = sparse.diags([1,-4,1], [-1,0,1], shape=(n,n))  # create the sparse matrix B
    A = sparse.block_diag([B]*n)  # create an n**2 by n**2 matrix with B as the diagonals
    A = A + sparse.diags([1,1],[-n,n],shape=(n**2,n**2))  # offset 1s are the same as block identity matrices
    return A

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    # initialize variables and empty times lists ###############################
    spsolve_times = []
    lasolve_times = []
    domain = np.arange(1,65)

    # use for-loop to time both methods of solving sparse matrix problems ######
    for n in domain:
        # initialize a random matrix A and vector b ############################
        A = prob5(n)
        b = np.random.random(n**2)

        # time spla.spsolve() ##################################################
        Acsr = A.tocsr()
        start_time = time.time()
        spla.spsolve(A, b)
        spsolve_times.append(time.time() - start_time)

        # time la.solve() ######################################################
        A = A.toarray()
        start_time = time.time()
        la.solve(A,b)
        lasolve_times.append(time.time() - start_time)

    # create plot ##############################################################
    plt.loglog(domain, spsolve_times, '.-', color = 'blue', linewidth = 1, markersize = 3, basex=2, basey=2, label = 'spsolve()') 
    plt.loglog(domain, lasolve_times, '.-', color = 'orange', linewidth = 1, markersize = 3, basex=2, basey=2, label = 'solve()')
    plt.legend(loc="upper left")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("time", fontsize=14)
    plt.show()