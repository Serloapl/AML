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

def PriorCompute(PointNum,RangeVal):
    #generate grid of weight pairs (w0,w1)
    w0 = np.linspace(-RangeVal, RangeVal, num=PointNum)
    w1 = np.linspace(-RangeVal, RangeVal, num=PointNum)
    Z = np.ones((len(w0), len(w0)))

    for i,x in enumerate(w0):
        for j,y in enumerate(w1):
            Z[i,j] = multivariate_normal([0, 0], [[0.2,0],[0,0.2]]).pdf(np.array([x,y]))

    return Z
def LikelihoodCompute(Z, point, PointNum, RangeVal):
    LocalSigma = 0.2
    #we redfine Z taking into account the new data point
    w0 = np.linspace(-RangeVal, RangeVal, num=PointNum)
    w1 = np.linspace(-RangeVal, RangeVal, num=PointNum)
    xi = point[0]
    yi = point[1]
    for i,x in enumerate(w0):
        for j,y in enumerate(w1):
            Z[i,j] = norm(x*xi+y, np.sqrt(LocalSigma)).pdf(yi) * Z[i,j]
    return Z
def pickDataPoint(w,sigma,mu):
    w0,w1 = w
    #get random x point in [-1:1]
    x = round(2*np.random.random_sample()-1, 2)
    epsilon = np.random.normal(mu,sigma)

    y = w0 * x + w1 + epsilon
    return (x, y)
def PointPlot(point):
    plt.plot(point[0],point[1],'ro')
def LikelihoodPlot(Z, RangeVal):
    img = plt.imshow(Z, cmap='jet', extent=(-RangeVal,RangeVal,-RangeVal,RangeVal),origin='lower')
    plt.xlabel('w1')
    plt.ylabel('w0')
    plt.show()
def SamplePlot(Z,PointNum, RangeVal):

    w0 = np.linspace(-RangeVal, RangeVal, num=PointNum)
    w1 = np.linspace(-RangeVal, RangeVal, num=PointNum)
    Zaux = Z.ravel()
    indices = np.argsort(Zaux)[::-1][:20]
    #invierto el orden para ir de mayor a menor y me quedo con los 20 mejores

    Wsamples =[]
    for i,(x,y) in enumerate(product(w0,w1)):
        for j, index in enumerate(indices):
            if i == index :
                    Wsamples.append((x,y))


    x = np.arange(-RangeVal, RangeVal, RangeVal/10)
    fig = plt.figure()
    for (w0, w1) in Wsamples:
        plt.plot(x, w0 * x + w1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()







#data parameters
w = [-1.3,0.5]
sigma = 0.3
mu = 0
#range
RangeVal = 2.0
PointNum = 100

Z =PriorCompute(PointNum,RangeVal)
#LikelihoodPlot(Z,RangeVal)
for i in xrange(25):
    point = pickDataPoint(w,sigma,mu)
    Z = LikelihoodCompute(Z, point, PointNum, RangeVal)


LikelihoodPlot(Z,RangeVal)
SamplePlot(Z,PointNum,RangeVal)