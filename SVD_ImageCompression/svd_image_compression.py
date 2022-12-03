# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""
# Marcelo Leszynski
# Math 321 Sec 005
# 11/09/20

import numpy as np
from scipy import linalg as la
from scipy import sparse
import math
from imageio import imread
from matplotlib import pyplot as plt


# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    lam, V = la.eig(A.conj().T @ A)
    sig = np.sqrt(lam)

    ranked_sig = np.argsort(sig)
    largest_indices_sig = ranked_sig[::-1]
    sig = sig[largest_indices_sig]
    V = V[:, largest_indices_sig]

    r = sum(sig > tol)
    r = np.count_nonzero(sig)

    sig = sig[:r]
    V = V[:,:r]
    U = A @ V / sig

    #print(np.allclose(U.T @ U, np.identity(5)))
    #print(np.allclose(U @ np.diag(sig) @ V.conj().T, A))
    #print(np.linalg.matrix_rank(A) == len(sig))

    return U, sig, V.conj().T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    # initializing and calculating necessare vars ##############################
    E = np.array([[1,0,0],[0,0,1]])
    theta = np.linspace(0, 2*math.pi, 200)
    x_vals = [math.sin(x) for x in theta]
    y_vals = [math.cos(x) for x in theta]
    unit_cir = np.vstack((x_vals, y_vals))
    U, sigma, Vh = la.svd(A)

    # plot first subplot #######################################################
    ax = plt.subplot(221)
    ax.plot(unit_cir[0,:], unit_cir[1,:])
    ax.plot(E[0,:], E[1,:], color='orange')
    ax.axis("equal")

    # plot second subplot ######################################################
    ax = plt.subplot(222)
    vhs = Vh @ unit_cir
    vhe = Vh @ E
    ax.plot(vhs[0,:], vhs[1,:])
    ax.plot(vhe[0,:], vhe[1,:], color='orange')
    ax.axis("equal")

    # plot third subplot #######################################################
    ax = plt.subplot(223)
    svhs = np.diag(sigma) @ (Vh @ unit_cir)
    svhe = np.diag(sigma) @ (Vh @ E)
    ax.plot(svhs[0,:], svhs[1,:])
    ax.plot(svhe[0,:], svhe[1,:], color='orange')
    ax.axis("equal")

    # plot fourth subplot ######################################################
    ax = plt.subplot(224)
    usvhs = U @ (np.diag(sigma) @ (Vh @ unit_cir))
    usvhe = U @ (np.diag(sigma) @ (Vh @ E))
    ax.plot(usvhs[0,:], usvhs[1,:])
    ax.plot(usvhe[0,:], usvhe[1,:], color='orange')
    ax.axis("equal")

    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    # error condition ##########################################################
    if s > np.linalg.matrix_rank(A):
        raise ValueError("s is too large")

    # calculate svd ############################################################
    U, e, Vh = la.svd(A)
    #U, e, Vh = compact_svd(A)

    # trim approximation matrices ##############################################
    Up = U[:,:s]
    Vhp = Vh[:s,:]
    ep = e[:s]
    return Up @ (np.diag(ep) @ Vhp), Up.size + ep.size + Vhp.size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    # check error condition ####################################################
    U, e, Vh = la.svd(A)
    if err <= e[-1]:
        raise ValueError("error bound is too small")

    # trim singular values that are too large ##################################
    ep = [i for i in e if i > err]
    ep = np.array(ep)
    s = len(ep)

    # trim svd approximation matrices ##########################################
    Up = U[:,:s]
    Vhp = Vh[:s,:]
    ep = e[:s]

    return Up @ (np.diag(ep) @ Vhp), Up.size + ep.size + Vhp.size
 

# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename) / 255

    # case where image is colored ##############################################
    if len(image.shape) != 2:  # image is colored 
        # approximate rbg layers ###############################################
        image_r, size_r = svd_approx(image[:,:,0], s)
        image_b, size_b = svd_approx(image[:,:,1], s)
        image_g, size_g = svd_approx(image[:,:,2], s)
        image_r, image_b, image_g = np.clip(image_r, 0, 1), np.clip(image_b, 0, 1), np.clip(image_g, 0, 1)
        image_p = np.dstack((image_r, image_b, image_g))

        # plot original image ##################################################
        ax = plt.subplot(121)
        ax.imshow(image)
        ax.axis('off')

        # plot compressed image ################################################
        ax = plt.subplot(122)
        ax.imshow(image_p)
        ax.axis('off')
        plt.suptitle('Difference: ' + str(image.size - (size_r+size_b+size_g)))

    # grayscale image case #####################################################
    else:
        #calculate approximation ###############################################
        image_p, size = svd_approx(image, s)
        image_p = np.clip(image_p, 0, 1)

        # plot original image ##################################################
        ax = plt.subplot(121)
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax = plt.subplot(122)

        # plot compressed image ################################################
        ax.imshow(image_p, cmap='gray')
        ax.axis('off')
        plt.suptitle('Difference: ' + str(image.size - size))

    plt.show()