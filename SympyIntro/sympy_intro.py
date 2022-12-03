# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
Marcelo Leszynski
Math 347 Section 3
02/25/21
"""

import numpy as np
import sympy as sy
from matplotlib import pyplot as plt

def prob1():
    """Return an expression for
        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).
    Make sure that the fractions remain symbolic.
    """
    # initialize symbols #######################################################
    x, y = sy.symbols('x, y')

    # create final expression ##################################################
    return sy.Rational(2,5) * sy.E ** (x**2 - y) * sy.cosh(x+y) + sy.Rational(3,7) * sy.log(x*y + 1)

def prob2():

    """Compute and simplify the following expression.
        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    # initialize x, i, j #######################################################
    x, i, j = sy.symbols('x i j')
    expr = j * (sy.sin(x) + sy.cos(x))

    # calculate the sum ########################################################
    sum = sy.summation(expr, (j, i, 5))

    # calculate the product ####################################################
    prod = sy.product(sum ,(i, 1, 5))

    return sy.simplify(prod)



# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    # initialize variables #####################################################
    x, y, n = sy.symbols('x y n')

    # form expression ##########################################################
    expr = x**n / sy.factorial(n)
    y2 = -y**2
    mac = sy.summation(expr, (n,0,N))
    mac_sub = mac.subs(x, y2)

    # plot the expression ######################################################
    domain = np.linspace(-2,2)
    f = sy.lambdify(y, mac_sub, "numpy")
    g = sy.lambdify(y, sy.exp(y2), "numpy")
    plt.plot(f(domain))
    plt.plot(g(domain))
    plt.show()


def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].

    """
    # initialize variables and form expression #################################
    x, y, r, theta = sy.symbols('x y r theta')
    rose_curve = 1 - (((x**2 + y**2)**sy.Rational(7,2)) + (18* x**5*y) - (60* x**3 *y**3) + (18*x* y**5)) / ((x**2 + y**2)**3)

    # convert to polar coordinates and lambdify ################################
    polar = rose_curve.subs({x:r*sy.cos(theta), y:r*sy.sin(theta)})
    simple = polar.simplify()
    solved = sy.solve(simple, r)

    # lambdify and plot ########################################################
    f = sy.lambdify(theta, solved[0], "numpy")
    domain = np.linspace(0,2*np.pi)
    plt.plot(f(domain)*np.cos(domain), f(domain)*np.sin(domain))
    plt.show()


def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    #first we define the matrix then take the determinant of A - lambdaI ##

    # define variables and matrix ##############################################
    x, y, lamb = sy.symbols('x y lamb')
    A = sy.Matrix([[x-y, x, 0], [x, x-y, x], [0, x, x-y]])

    # calculate the determinant ################################################
    det = sy.det(A -lamb*sy.eye(3))
    eig_vals = sy.solve(det, lamb)

    # calculate and return eigenvalue/vector mappings
    dic = {}
    for eig_val in eig_vals:
        dic[eig_val] = (A-eig_val*sy.eye(3)).nullspace()

    return dic


def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """

    #first we have the expression then we lambdify it ##

    # initialize and compute the expression ####################################
    x, y = sy.symbols('x y')
    min = []
    max = []
    p = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100

    # lambdify the expression and compute differentials ########################
    px = sy.lambdify(x, p, "numpy")
    d1 = sy.diff(p, x)
    d2 = sy.diff(p, x, x)
    d2 = sy.lambdify(x, d2, "numpy")

    # find critical points and categorize them #################################
    crits = sy.solve(d1,x)
    for i in crits:
        if d2(i) > 0:
            min.append(i)
        elif d2(i) < 0:
            max.append(i)

    mins = np.array(min)
    maxs = np.array(max)

    # plot the values ##########################################################
    domain = np.linspace(-5,5)
    plt.plot(domain, px(domain), label="p(x)")
    plt.scatter(mins, px(mins), label="minima")
    plt.scatter(maxs, px(maxs), label="maxima")
    plt.legend(loc='upper left')
    plt.show()

    return set(mins), set(maxs)


def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    #first we get the expression then we integrate ##

    # initialize variables, calculate the expression ###########################
    x, y, z, p, t, phi, r = sy.symbols('x y z row theta phi r')
    f = (x**2 + y**2)**2
    h = sy.Matrix([p*sy.sin(phi)*sy.cos(t), p*sy.sin(phi)*sy.sin(t), p*sy.cos(phi)])
    J = -sy.simplify(h.jacobian([p, t, phi]))

    integrand = sy.simplify(f.subs({x:h[0], y:h[1], z:h[2]}) * J.det())
    final_val = sy.integrate(integrand, (p,0,r), (t,0,2*sy.pi), (phi,0,sy.pi))

    S = sy.lambdify(r, final_val, "numpy")

    # plot values ##############################################################
    domain = np.linspace(0,3)
    plt.plot(domain,S(domain))
    plt.show()

    return S(2)
