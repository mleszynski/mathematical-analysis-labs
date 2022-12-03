# solutions.py
"""Volume 1: The Page Rank Algorithm.
Marcelo Leszynski
Math 347 Section 3
03/22/21
"""

import numpy as np
import networkx as nx
from scipy import linalg as la


# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        # get correct labels ###################################################
        self._n = len(A)
        if labels is None:
            self._labels = list(range(self._n))
        else:
            self._labels = labels

        # make sure each node has a label ######################################
        if len(self._labels) != self._n:
            raise ValueError("Number of labels is not equal to the number of nodes in the graph")

        A_hat = np.zeros((self._n,self._n))
        for j in range(self._n):
            # check that there are no sinks in A ###############################
            if np.sum(A[:,j]) == 0:
                A[:,j] = 1

            A_hat[:,j] = A[:,j] / np.sum(A[:,j])

        self._A_hat = A_hat


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # use the linear method Ax = b #########################################
        A = np.identity(self._n) - epsilon * self._A_hat
        b = np.ones(self._n) * (1-epsilon) / self._n
        p = la.solve(A,b)

        return {self._labels[i]:p[i] for i in range(self._n)}


    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # calculate matrix B ###################################################
        E = np.ones((self._n,self._n))
        E_scalar = (1-epsilon) / self._n
        B = (self._A_hat*epsilon) + (E*E_scalar)

        # compute the eigenvector ##############################################
        vals, vecs = la.eig(B, right=True)
        close = np.isclose(vals,1)
        index = np.where(close == True)[0][0]

        _p = vecs[:,index]
        p = _p / _p.sum()

        return {self._labels[i]:p[i] for i in range(self._n)}



    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # use the iterative solution ###########################################
        ones = np.ones(self._n)
        ones_scaled = (1-epsilon) * ones / self._n
        eps_A_hat = epsilon * self._A_hat
        p = ones / self._n

        # iterate through ######################################################
        for i in range(maxiter):
            p_1 = eps_A_hat @ p + ones_scaled
            if la.norm(p_1-p) < tol:
                p = p_1
                break
            p = p_1

        return {self._labels[i]:p[i] for i in range(self._n)}


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.
    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    # sort by keys then sort by values greates to least ########################
    key_dict = {i:d[i] for i in sorted(d.keys(), reverse=False)}
    return sorted(key_dict, key=key_dict.get, reverse=True)


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    # access the file ##########################################################
    with open(filename, "r") as myfile:
        lines = myfile.read().strip().split("\n")

    # make list of unique ID labels ############################################
    IDs = set()
    for line in lines:
        IDs.update(set(line.split("/")))

    # create a dictionary to know the row and column index #####################
    labels = sorted(IDs) #convert strings to integers
    dict = {labels[i]:i for i in range(len(labels))}

    # insert a one into linked website indices #################################
    A = np.zeros((len(IDs),len(IDs)))
    for line in lines:
        websites = line.split("/")

        col = dict.get(websites[0])
        for i in range(1,len(websites)):
            row = dict.get(websites[i])
            A[row,col] = 1

    # make a DiGraph object and use it to solve ################################
    DG = DiGraph(A,labels)
    return get_ranks(DG.itersolve(epsilon))


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    # access file and initialize arrays ########################################
    with open(filename, "r") as myfile:
        lines = myfile.read().strip().split("\n")

    IDs = set()
    for line in lines[1:]:
        IDs.update(set(line.split(",")))

    labels = sorted(IDs)
    dict = {labels[i]:i for i in range(len(labels))}

    A = np.zeros((len(IDs),len(IDs)))
    # update win/loss graph ####################################################
    for line in lines[1:]:
        teams = line.split(",")

        winner = dict.get(teams[0])
        loser = dict.get(teams[1])

        A[winner,loser] += 1

    DG = DiGraph(A,labels)
    return get_ranks(DG.itersolve(epsilon))



# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    # access the file ##########################################################
    DG = nx.DiGraph()
    with open(filename, "r", encoding="utf-8") as myfile:
        movies = myfile.read().strip().split("\n")

    # update nodes using actor info ############################################
    for movie in movies:
        actors = movie.split("/")[1:]

        for i in range(len(actors)):
            for j in range(i+1,len(actors)):
                if not DG.has_edge(actors[j],actors[i]):
                    DG.add_edge(actors[j], actors[i], weight=1)
                else:
                    DG[actors[j]][actors[i]]["weight"] += 1

    return get_ranks(nx.pagerank(DG, alpha=epsilon))