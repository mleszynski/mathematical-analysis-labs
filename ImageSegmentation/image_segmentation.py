# image_segmentation.py
"""Volume 1: Image Segmentation.
Marcelo Leszynski
Math 345 Sec 005
11/06/20
"""

import numpy as np
import math
from scipy import linalg as la
from scipy import sparse
from imageio import imread
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    # calculate the degree matrix
    D = np.diag(np.sum(A, axis = 0))
    # return the laplacian matrix
    return D - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # create a laplacian of A and find its eigenvalues #########################
    L = laplacian(A)
    eigens = np.sort(np.real(la.eigvals(L)))  # sort to make finding algebraic connectivity easier
    num_connected = 0
    
    # calculate number of connected nodes ######################################
    for i in eigens:
        if i < tol:
            num_connected += 1

    return num_connected, eigens[1]


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self._original = imread(filename) / 255
        self._is_grayscale = True
        self._num_rows = np.shape(self._original)[0]
        self._num_cols = np.shape(self._original)[1]
        if len(np.shape(self._original)) != 2:
            self._is_grayscale = False
        if self._is_grayscale:
            self._brightness = np.ravel(self._original)
        else:
            self._brightness = np.ravel(self._original.mean(axis = 2))



    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self._is_grayscale:
            plt.imshow(self._original, cmap='gray')
        else:
            plt.imshow(self._original)
        plt.axis('off')
        plt.show()


    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        # initialize A and D ###################################################
        dim_A = self._num_rows * self._num_cols
        A = sparse.lil_matrix((dim_A, dim_A), dtype='float')
        D = np.array([0] * dim_A, dtype='float')

        # compute weights and degrees ##########################################
        for i in range(0, dim_A):
            neighbors, distances = get_neighbors(i, r, self._num_rows, self._num_cols)
            # calculate weights ################################################
            weights = [math.exp(-abs(self._brightness[i]-self._brightness[neighbors[j]]) / sigma_B2 - distances[j] / sigma_X2) for j in range(len(neighbors))]
            A[i, np.array(neighbors)] = weights
            # calculate D ######################################################
            D[i] = np.sum(weights)

        return A.tocsc(), D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        # construct Laplacian and D^(-1/2) #####################################
        L = sparse.csgraph.laplacian(A)
        processed_D = sparse.diags(D)
        processed_D = processed_D.power(-1/2)  # TODO: seems sketch

        # perform DLD operation, calculate eigenvector, and construct mask #####
        DLD = processed_D @ L @ processed_D
        e_vals, e_vectors = eigsh(DLD, k=2, which='SM')
        mask = e_vectors[:,1].reshape((self._num_rows, self._num_cols)) > 0

        return mask
        
    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        # create the image mask ################################################
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)

        # plot a grayscale image ###############################################
        if self._is_grayscale:
            plt.subplot(131)
            plt.imshow(self._original, cmap='gray')
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(self._original*mask, cmap='gray')
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(self._original*(~mask), cmap='gray')
            plt.axis('off')

        # plot a colored image #################################################
        else:
            shape_size = [mask] * np.shape(self._original)[2]
            mask = np.dstack(shape_size)
            plt.subplot(131)
            plt.imshow(self._original)
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(self._original*mask)
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(self._original*(~mask))
            plt.axis('off')
        plt.show()