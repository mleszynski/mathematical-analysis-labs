# profiling.py
"""Python Essentials: Profiling.
Marcelo Leszynski
Math 347 Section 3
19 February 2021
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import math as m
from numba import jit
from matplotlib import pyplot as plt
import time

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    # read file and initialize data ############################################
    with open(filename, 'r') as my_file:
        my_data = [[int(n) for n in line.split()] for line in my_file.readlines()]

    def fast_path():
        for i in range(1,len(my_data)):
            # iterate from the bottom up #######################################
            for j in range(len(my_data[-i-1])):
                sum = max(my_data[-i][j], my_data[-i][j+1])
                my_data[-i-1][j] += sum

        return my_data[0][0]

    return fast_path()

# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors. ##
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    # initialize variables #####################################################
    primes_list = [2]
    current = 3
    while len(primes_list) < N:
        isprime = True
        # check for divisors ###################################################
        for i in range(2, current):     
            if i > m.sqrt(current):
                break
            if current % i == 0:
                isprime = False
                break
        if isprime:
            primes_list.append(current)
        # check next prime, skipping even numbers ##############################
        current += 2 
    return primes_list

# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    # np.newaxis computes additions and subtractions efficiently ###############
    return np.argmin(np.linalg.norm(A - x[:,np.newaxis], axis=0))

# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as myfile:
        names = sorted(myfile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    # initialize variables and read file #######################################
    with open(filename, 'r') as myfile:
        names = sorted(myfile.read().replace('"', '').split(','))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # creating a dictionary makes value lookup easier ##########################
    my_dic = {l:i+1 for i,l in enumerate(alphabet)}
    total=0

    # iterate using an enumerate for efficiency ################################
    for i,name in enumerate(names):
        name_val = 0
        for letter in name:
            name_val += my_dic[letter]
        total += (i+1) * name_val
    return total

# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    f_1 = 0
    f_2 = 0

    while True:
        if f_2 == 0: # to not miss the first two fib numbers ##
            f_2 = 1
            yield f_2
        f_next = f_1 + f_2
        f_1 = f_2
        f_2 = f_next
        yield f_next

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    # use an enumerate for efficient calculation ###############################
    for i, f in enumerate(fibonacci()):
        # length of string == number of digits #################################
        if len(str(f)) >= N: 
            return i + 1

# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    # create all lists
    all_ints = [i for i in range(2, N+1)]

    while len(all_ints) > 0:
        temp_prime = all_ints[0]
        for i, num in enumerate(all_ints):
            # clean up duplicate primes ########################################
            if num % temp_prime == 0: 
                all_ints.pop(i)
        yield temp_prime

# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temp_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temp_array[j] = total
            product[i] = temp_array
    return product

@jit # use jit decoration ######################################################
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temp_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temp_array[j] = total
            product[i] = temp_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    #initialize variables ######################################################
    mat_time = []
    mat_numb_time = []
    lin_alg_time = []
    m_m = [2**i for i in range(2,8)]
    A = np.random.random(size=(2,2))
    # compile matrix_power_numba ###############################################
    comp = matrix_power_numba(A, n) 

    for m in m_m:
        # initialize a test matrix #############################################
        A = np.random.random(size=(m,m))

        # time matrix power function ###########################################
        start = time.time()
        a = matrix_power(A,n)
        end = time.time()
        mat_time.append(end - start)

        # time matrix power numba function #####################################
        start = time.time() 
        b = matrix_power_numba(A,n)
        end = time.time()
        mat_numb_time.append(end - start)

        # time linalg matrix power function ####################################
        start = time.time()
        c = np.linalg.matrix_power(A,n)
        end = time.time()
        lin_alg_time.append(end - start)

    # create plot ##############################################################
    xvals= np.arange(2,8)
    plt.title("Matrix Power Times")
    plt.loglog(xvals,mat_time,'b', label="Matrix Power")
    plt.loglog(xvals,mat_numb_time, 'g', label="Matrix Power Numba")
    plt.loglog(xvals,lin_alg_time,'r', label="NP Matrix Power")
    plt.legend()
    plt.xlabel("Size 2**i")
    plt.ylabel("Time (s)")
    plt.show()