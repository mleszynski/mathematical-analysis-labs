# linear_transformations.py
"""Volume 1: Linear Transformations.
Marcelo Leszynski
Math 345 Sec 005
09/29/20
"""

import numpy as np
from matplotlib import pyplot as plt
from random import random
import time

# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    transform = np.array([[a,0], [0,b]])
    return np.matmul(transform, A)

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    transform = np.array([[1, a], [b, 1]])
    return np.matmul(transform, A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    transform = np.array([[(a**2-b**2)/(a**2+b**2), (2*a*b)/(a**2+b**2)], [(2*a*b)/(a**2+b**2), (b**2-a**2)/(a**2+b**2)]])
    return np.matmul(transform, A)

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    transform = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(transform, A)

# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    #  Initializing Variables ##################################################
    times = np.linspace(0, T, 1000)  # generate a list of times
    earth_0 = np.array([x_e, 0])
    moon_0 = np.array([x_m, 0])
    earth = np.copy(earth_0)
    moon = np.copy(earth_0)
    #  Setting Up Graph ########################################################
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #  Loop ####################################################################
    for time in times:
        ax.plot(earth[0], earth[1], 'bo', markersize=1.0)  # Plot current earth location
        ax.plot(moon[0], moon[1], 'o', color='orange', markersize=1.0)  # Plot current moon location
        earth = rotate(earth_0, time*omega_e)  # Rotate the earth
        moon = rotate(moon_0-earth_0, time*omega_m)  # Rotate the moon relative to the origin
        moon += earth  # Translate the moon to location relative to earth

    #  Plot Results ############################################################
    ax.set_aspect("equal")
    plt.show()



def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    #  Initialize Variables ####################################################
    domain = 2**np.arange(1,9)
    times_matmat = []
    times_matvec = []

    #  Loop ####################################################################
    for n in domain:
        A = random_matrix(n)  # Generate random matrix
        B = random_matrix(n)  # Generate random matrix
        x = random_vector(n)  # Generate random vector

        #  Time Mat x Mat ######################################################
        start = time.time()  # Get initial time
        matrix_matrix_product(A, B)
        times_matmat.append(time.time() - start)  # Calculate time diff

        #  Time Mat x Vec ######################################################
        start = time.time()  # Get initial time
        matrix_vector_product(A, x)
        times_matvec.append(time.time() - start)  # Calculate time diff

    #  Plot Results ############################################################
    plt.subplot(122)
    plt.title("Matrix x Matrix")
    plt.plot(domain, times_matmat, '.-', color='orange', linewidth=2, markersize=15)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("", fontsize=14)
    plt.subplot(121)
    plt.title("Matrix x Vector")
    plt.plot(domain, times_matvec, 'g.-', color='blue', linewidth=2, markersize=15)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.tight_layout(pad=3)
    plt.show()

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    #  Initialize Variables ####################################################
    domain = 2**np.arange(1,9)
    matrix_matrix_p = []
    matrix_vector_p = []
    matrix_matrix_d = []
    matrix_vector_d = []

    #  Loop ####################################################################
    for n in domain:
        A = random_matrix(n)  # Generate random matrix
        B = random_matrix(n)  # Generate random matrix
        x = random_vector(n)  # Generate random vector

        #  Time Mat x Mat Product ##############################################
        start = time.time()
        matrix_matrix_product(A, B)
        matrix_matrix_p.append(time.time() - start)  # Calculate time diff

        #  Time Mat x Vec Product ##############################################
        start = time.time()
        matrix_vector_product(A, x)
        matrix_vector_p.append(time.time() - start)  # Calculate time diff

        #  Time Mat x Mat Dot ##################################################
        start = time.time()
        np.dot(A, B)
        matrix_matrix_d.append(time.time() - start)  # Calculate time diff

        #  Time Mat x Vec Dot ##################################################
        start = time.time()
        np.dot(A, x)
        matrix_vector_d.append(time.time() - start)

    #  Plot Results ############################################################
    ax1 = plt.subplot(121)
    ax1.plot(domain, matrix_matrix_p, '.-', color='orange', linewidth=2, markersize=15, label = "Matrix-Matrix")
    ax1.plot(domain, matrix_vector_p, '.-', color='blue', linewidth=2, markersize=15, label = "Matrix-Vector")
    ax1.plot(domain, matrix_matrix_d, '.-', color='green', linewidth=2, markersize=15, label = "Matrix-Matrix (numpy)")
    ax1.plot(domain, matrix_vector_d, '.-', color='purple', linewidth=2, markersize=15, label = "Matrix-Vector (numpy)")
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(122)
    ax2.loglog(domain, matrix_matrix_p, '.-', color='orange', basex=2, basey=2, lw=2, label = "Matrix-Matrix")
    ax2.loglog(domain, matrix_vector_p, '.-', color='blue', basex=2, basey=2, lw=2, label = "Matrix-Vector")
    ax2.loglog(domain, matrix_matrix_d, '.-', color='green', basex=2, basey=2, lw=2, label = "Matrix-Matrix (numpy)")
    ax2.loglog(domain, matrix_vector_d, '.-', color='purple', basex=2, basey=2, lw=2, label = "Matrix-Vector (numpy)")
    ax2.legend(loc="upper left")
    plt.show()
