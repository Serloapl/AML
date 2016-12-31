import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import multivariate_normal


def LikelihoodPlot(Z, RangeVal):
    img = plt.imshow(Z, cmap='jet', extent=(0,RangeVal,0,RangeVal),origin='lower')
    plt.xlabel('tao')
    plt.ylabel('mu')
    plt.show()
def ComputePosterior(muN,lambdaN,aN,bN):
    # generate grid of weight pairs (w0,w1)
    taos = np.linspace(0, 6, num=300)
    mus = np.linspace(0, 6, num=300)

    q_mu = norm(muN, (lambdaN ** -0.5)).pdf(mus)[:,None]
    q_taos = gamma.pdf(taos, aN, scale=bN ** -1)[None,:]

    Z = np.dot(q_mu,q_taos)

    #optional loop calculation (SUPER SLOW!)
    #Z = np.ones((len(taos), len(mus)))
    #for i, mu in enumerate(mus):
    #    for j, tao in enumerate(taos):
    #        Z[i, j] = norm(muN, (lambdaN**-0.5)).pdf(mu)*gamma.pdf(tao,aN,scale=bN**-1)

    return Z
def GenerateData(NumPoints, mu, tao):
    # data = []
    # for i in range(1000):
    #     # mu_aux = np.random.normal(muData,lambdaData**-0.5,1)
    #     # tao_aux = gamma.rvs(aData,scale=bData**-1,size=1)
    #     data.append(np.random.normal(muData, lambdaData ** -0.5, 1))

    data = np.array(np.random.normal(mu, tao ** -0.5, NumPoints))
    return data




#data generation parameters
muData = 2
taoData = 3

data = GenerateData(500, muData, taoData)
N = len(data)
meanData = np.mean(data)

#priors parameters
mu0 = 1
lambda0 = 2
a0 = 2
b0 = 0.5

#Z = ComputePosterior(mu0,lambda0,a0,b0)
#LikelihoodPlot(Z,6)

#initial tao guess
expTao = 1

for i in range(100):
    muN = (lambda0*mu0 + N*meanData)/(lambda0 + N)
    lambdaN = (lambda0 + N)*expTao

    #now compute aN,bN
    aN = a0 + N/2

    #bN developing and rearranging
    Emu2=(lambdaN**-1) + muN**2
    sumx = sum(data)
    sumx2 = sum(data**2)

    bN = b0 + 0.5*(sumx2 + lambda0*(mu0**2) - (2*sumx + 2*lambda0*mu0)*muN + (N + lambda0)*Emu2 )

    #directly computing using linespace
    # mus = np.linspace(-6, 6, num=300)
    # factor = [sum([(d-mu)**2 for d in data]) + lambda0*(mu - mu0)**2 for mu in mus ]
    # probs = [norm(muN, (lambdaN**-0.5)).pdf(mu) for mu in mus ]
    # bN = b0 + 0.5*np.mean(np.multiply(factor,probs))

    #recompute expTao
    r = gamma.rvs(aN,scale=bN**-1, size=1000)
    expTao = np.mean(r)

    #alternatively expTao calculated following Bishop approximation
    expTao=aN/bN

print muN,lambdaN,aN,bN

#Posterior plot
Z = ComputePosterior(muN,lambdaN,aN,bN)
LikelihoodPlot(Z,6)


#derived Posterior parameters
muP = (lambda0*mu0 + N*meanData)/(lambda0+N)
lambdaP = lambda0 + N
aP = a0 + N/2
bP = b0 + 0.5*sum((data-meanData)**2) + (lambda0*N*(meanData-mu0)**2)/(2*(lambda0+N))
print muP, lambdaP, aP, bP

#Posterior plot
Z = ComputePosterior(muP,lambdaP,aP,bP)
LikelihoodPlot(Z,6)