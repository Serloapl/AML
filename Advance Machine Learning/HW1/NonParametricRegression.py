__author__ = 'Sergio'
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sys import platform as _platform
from scipy.spatial.distance import cdist, pdist
import pylab as pb


def generateData(NumPoints):
    X = np.linspace(-np.pi, np.pi, NumPoints)
    epsilon = np.random.normal(0, np.sqrt(0.5), len(X))
    Y = np.sin(X)+epsilon
    return (X,Y)

def kernel(X, Y, l):
    return np.exp(-cdist(X, Y, 'sqeuclidean')/(l*l))

def computePosterior(xStar, X, Y,l):

    #create local copy
    Xstar = xStar
    #define second dimension (undefined in Python)
    X = X[:,None]
    Xstar = Xstar[:,None]
    t = Y[:,None]

    #Compute k
    k = kernel(Xstar,X,l)

    #Compute C and inverse it
    I = (0.2**2) * np.identity(len(X))
    C = kernel(X,X,l)+ I
    Cinv = np.linalg.inv(C)

    #compute mu
    mu = np.dot(np.dot(k,Cinv),t)
    #compute sigma
    c = kernel(Xstar, Xstar,l)
    sigma = c- np.dot(np.dot(k,Cinv),np.transpose(k))

    return mu, sigma

def plotSamplePos(mu,sigma,x, NumSamples):

    mu = np.reshape(mu,(len(x),))
    x = x[:,None]
    Z = np.random.multivariate_normal(mu,np.nan_to_num(sigma),NumSamples)
    pb.figure() # open new plotting window?
    pb.plot(X,Y,'ro')
    for i in range(NumSamples):
        pb.plot(x[:],Z[i,:])
    pb.show()

def plotPosterior(mu,sigma,x):

    #plot data points
    plt.plot(X, Y,'ro')
    #plot perfect sin(x) form
    plt.plot(x,np.sin(x), color = 'green')

    #plot average
    mu = np.reshape(mu, (len(x),))
    plt.plot(x,mu, color = 'blue')

    #plot upper and low bond
    upper = mu + 2*np.sqrt(sigma.diagonal())
    lower = mu - 2*np.sqrt(sigma.diagonal())
    ax = plt.gca()
    ax.fill_between(x, upper, lower, facecolor='green', interpolate=True, alpha=0.1)
    #plt.title(title)
    plt.show()

def computeGPSamples(l,NumSamples = 10):

    NumPoints = 800
    # and reshape X to make it n*D (we define the 2nd dimension)
    X = np.linspace(-2, 2,NumPoints)[:,None]
    #mu is a zero vector length the same as the number of data points
    mu = np.zeros(NumPoints) # vector of the means
    K = kernel(X,X,l)
    Z = np.random.multivariate_normal(mu,K,NumSamples)

    return X,Z

def plotSamples(X,Z,l,NumSamples = 10):
    print np.shape(X)
    print np.shape(Z)
    #plotting
    pb.figure()
    for i in range(NumSamples):
        pb.plot(X,Z[i,:]) # plot a slice of Z, which is a realization
    pb.title('l : '+str(l))
    pb.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#range
l = 2
NumSamples = 10
NumDataPoints = 7
#Compute Prior and plot some samples
#[X,Z] = computeGPSamples(l,NumSamples)
#plotSamples(X,Z,l,NumSamples)

#Generate data
X,Y = generateData(NumDataPoints)
#Generate axis
axis = np.linspace(-2*np.pi, 2*np.pi, 800)
#Compute Posterior
mu, sigma = computePosterior(axis, X, Y, l)
#plot Posterior samples
plotSamplePos(mu,sigma,axis, NumSamples)
plotPosterior(mu,sigma,axis)



