import numpy as np
import math
from scipy import linalg
import matplotlib.pyplot as plt


###this function calculates the B-spline basis functions recursively (see class notes)
###taken from https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline
def B(x, kk, ii, tt):
    """
    Parameters
    ----------
    x : float
        x-value on the base interval.
    kk : int
       B-spline degree
    ii : int
        Index for the knot vector.
    tt : array
        Knots.

    Returns
    -------
    float
        B-spline basis function evaluated at x.

    """
    if kk == 0:
        return 1.0 if tt[ii] <= x < tt[ii + 1] else 0.0
    if tt[ii + kk] == tt[ii]:
        c1 = 0.0
    else:
        c1 = (x - tt[ii]) / (tt[ii + kk] - tt[ii]) * B(x, kk - 1, ii, tt)
    if tt[ii + kk + 1] == tt[ii + 1]:
        c2 = 0.0
    else:
        c2 = (tt[ii + kk + 1] - x) / (tt[ii + kk + 1] - tt[ii + 1]) * B(x, kk - 1, ii + 1, tt)
        
    return c1 + c2


###this function calculates the point value of the spline at a given point "x" (see class notes)
###taken from https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline
def bspline(x, tt, c, kk):
    """
    Parameters
    ----------
    x : float
        x-value on the base interval.
    tt : array
        Knots.
    c : array
        Spline coefficients.
    kk : int
        B-spline degree

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    nn = len(tt) - kk - 1
    
    assert (nn >= kk + 1) and (len(c) >= nn)
    
    return sum(c[ii] * B(x, kk, ii, tt) for ii in range(nn))

def get_points_from_branch(B_set, q):
    """
    Parameters
    ----------
    B_set : list
        Set of branches B.
    q : int
        Index of a branch B in B_set.

    Returns
    -------
    p : 2d array
        Points of a branch B.
    """
    
    p = np.zeros((2, len(B_set[q])))
    
    for qq in range(len(B_set[q])):
        p[0][qq] = B_set[q][qq][1]
        p[1][qq] = B_set[q][qq][0]
        
    return p

def find_control_point_locations(Np, Nq, M):
    """
    Parameters
    ----------
    Np : int
        Number of sample points (# of data points) for B-spline fitting.
    Nq : int
        Number of control points for the B-spline fit.
    M : 2d array
        Basis matrix for cubic B-spline fit.
        
    Returns
    -------
    Ntwo : 2d array
        Coordinates for the best control points.
    """
    
    Ntwo = np.zeros((Np, Nq))
    ## Calculate the best control point locations
    # This loop calculates basis functions (N) of Bspline and solves P = NQ where Q is control points (Q in the loop)
    # and P is the sample data points
    for i in range(Np):  # iterate through all sample data points

        # float
        index_f = Nq * ((i) / Np)

        # int
        index_i = math.floor(index_f)  # converts float to integer
        t_t = index_f - index_i  # finds interval for knot segment

        # Q-1 Q0 Q1 Q2
        indexV = M[0] - M[0]  # create 1x4 array of zeros

        for j in range(len(M[0])):
            indexV[j] = index_i + j  # iterate through control points
        indexV[np.argwhere(indexV < 1)] = 1  # ensure indices are between 1 and # of control points
        indexV[np.argwhere(indexV > Nq)] = Nq  # ensure indices are between 1 and # of control points

        tv = [t_t ** 3, t_t ** 2, t_t,
              1]  # matrix form of difference in knot segment size to be multiplied by the basis matrix

        # multiply knot segment size and basis matrices, update the basis functions
        temp = np.dot(tv, M) / 6
        Ntwo[i, indexV[0] - 1] = Ntwo[i, indexV[0] - 1] + temp[0]
        Ntwo[i, indexV[1] - 1] = Ntwo[i, indexV[1] - 1] + temp[1]
        Ntwo[i, indexV[2] - 1] = Ntwo[i, indexV[2] - 1] + temp[2]
        Ntwo[i, indexV[3] - 1] = Ntwo[i, indexV[3] - 1] + temp[3]
        
    return Ntwo


def compute_b_spline_points(B_set, q, p, resolution=15, nCP=5):
    """
    Parameters
    ----------
    B_set : list
        Each element is a list of (x,y) tuple coordinates for branch points
    q: int
        Index of the branch B in B_set that we will fit.
    p: 2d array
        The original branch points to be fitted.
    resolution : int, optional
        The number of points used in interpolation. More points yields
        more accurate results. The default is 15.
    nCP : int, optional
        The number of control points for a B-spline, n+1. The default is 5.

    Returns
    -------
    Px : array
        B-spline fit x-coordinates.
    Py : array
        B-spline fit y-coordinates.
    """
    n = nCP - 1  # order of b-spline, n is based on n = (number of control points) - 1
    k = 3  # order of B-spline
    T = k + n + 2  # number of knots in the knot vector
    
    # Create the knot vector, t
    t = np.linspace(0, T - 1, T)  # knot vector with uniform spacing from 0 to T-1 with T points
    tFront = np.zeros((1, k+1))  # make first k entries = 0 for end point interpolation
    tMid = np.linspace(2, T - k - 4, T - k * 2 - 2)  # uniform spacing of middle knots
    tRear = np.zeros((1, k+1)) + T - k - 2  # make last k entries = T for end point interpolation
    t = np.concatenate((tFront, tMid, tRear), axis=None)  # concatenate t, knot vector

    # Create basis functions
    uVals = np.linspace(t[0], t[-1], resolution)
    Px = 0 * uVals
    Py = 0 * Px

    # Calculate the control point locations
    Np = len(p[0])  # number of sample points = # of data points
    Nq = T - k - 1 # number of control points 

    if Nq > Np:  # limit number of control points to be no greater than number of data points
        Nq = Np

    # Basis matrix
    M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]])  # THIS ONLY WORKS FOR CUBIC!!!!

    Ntwo = find_control_point_locations(Np, Nq, M)

    invNN_Nt = np.dot(linalg.inv(np.dot(Ntwo.T, Ntwo)), Ntwo.T)  # solve equation NQ = P

    # Control points
    Q = np.dot(p, invNN_Nt.T)

    for u in range(resolution):
        
        ux = uVals[u]
        Px[u] = bspline(ux, t, Q[0], k)
        Py[u] = bspline(ux, t, Q[1], k)

    endpoints = 1  # change this to 0 if end point interpolation is not desired
    remove_last = 0  # change this to 1 if you want to delete the last entry of interpolated points (last point
                        # will be (0,0) if endpoints = 0 in line above
    if endpoints == 1:
        Px[0] = p[0][0]
        Px[-1] = p[0][-1]
        Py[0] = p[1][0]
        Py[-1] = p[1][-1]
    if remove_last == 1:
        Px = np.delete(Px, -1)
        Py = np.delete(Py, -1)
    
    return Px, Py

def plot_bspline_result(B_set, image_name, bw_img, dpi=200):
    """
    Parameters
    ----------
    B_set : list
        Set of branches B to be fitted to B-splines.
    image_name : string
        Name of the image for plotting.
    bw_img : 2d array
        Skeletonized binary image
    dpi : int, optional
        Dots per image for the plot. The default is 200.

    Returns
    -------
    None.
    """

    plt.figure(dpi=dpi) # Initialize empty figure
    
    for q in range(len(B_set)):
        
        p = get_points_from_branch(B_set, q)
        
        if len(p[0]) <= 4:
            continue

        Px, Py = compute_b_spline_points(B_set, q, p)
        
        '''
        PLOTTING
        '''
        c1 = "deepskyblue"
        if q == 0:
            plt.plot(p[0], p[1], 's', label="Original", markersize=2, color=c1)
            plt.plot(Px, Py, '-r.', label="Calculated")
        else:
            plt.plot(p[0], p[1], 's', markersize=2, color=c1)
            plt.plot(Px, Py, '-r.')
    
    # Complete the plot
    plt.title("BSpline curve fitting: " + image_name)
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid()
    plt.imshow(bw_img, cmap='gray')
    plt.show()