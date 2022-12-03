# qr_decomposition.py
"""Volume 1: The QR Decomposition.
Marcelo Leszynski
Math 345 Sec 005
10/23/20
"""
import numpy as np
from scipy import linalg as la


# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    # Initialization of necessary variables ####################################
    m, n = np.shape(A)
    Q = np.copy(A)
    s = (n,n)
    R = np.zeros(s)

    # Graham Schmidt process ###################################################
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i + 1, n):
            R[i,j] = (Q[:,j].T) @ Q[:,i]
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q, R

# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q, R = la.qr(A, mode = "economic")
    return np.abs(np.prod(np.diag(R)))  # returning the trace


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    # initializing necessary variables #########################################
    n = len(b)
    x = np.zeros(n)
    Q, R = la.qr(A, mode = "economic")
    y = (Q.T)@(b)

    # performing the backwards substitution ####################################
    for i in range(n):
        z = 0
        for j in range(n):
            z += x[j] * R[n-1-i, j]
        x[n-1-j] = (y[n-1-i] - z) / R[n-1-i, n-1-i]
    
    return x


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    # initializing necessary variables #########################################
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.eye(m)
    sign = lambda x : 1 if x >= 0 else -1

    # perform the householder algorithm ########################################
    for k in range(n):
        u = np.copy(R[k:,k])
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u / la.norm(u)
        R[k:,k:] = R[k:,k:] - 2 * (np.outer(u, u.T @ R[k:,k:]))
        Q[k:,:] = Q[k:,:] - 2 * (np.outer(u, u.T @ Q[k:,:]))
    
    return Q.T, R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    # initializing necessary variables #########################################
    m, n = np.shape(A)
    H = np.copy(A)
    Q = np.eye(m)
    sign = lambda x : 1 if x >= 0 else -1

    # perform the hessenberg algorithm #########################################
    for k in range(n-2):
        u = np.copy(H[k+1:,k])
        u[0] = u[0] + sign(u[0]) * la.norm(u)
        u = u / la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2 * (np.outer(u, u.T @ H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2 * (np.outer(H[:,k+1:] @ u, u.T))
        Q[k+1:,:] = Q[k+1:,:] - 2 * (np.outer(u, u.T @ Q[k+1:,:]))
    
    return H, Q.T
