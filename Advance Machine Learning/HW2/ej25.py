import numpy as np
import matplotlib.pyplot as plt
import random as rdm
from scipy.stats import norm

def rollBiasedDice(probs):
    roll = rdm.random() # in [0,1)
    acumulated = 0.
    for outcome,prob in enumerate(probs):
        if(roll<=prob+acumulated):
            return outcome+1
        acumulated+=prob

def ComputeNextTable(NextTable):
    num = rdm.randint(1,4)
    if(num != 1):
        return 1-NextTable
    else:
        return NextTable

def ComputeAlphaAndPhi(N, alpha, phi):

    alpha = (N.sum(axis=Maxis).sum(axis=Daxis))/(N.sum(axis=Laxis).sum(axis=Maxis).sum(axis=Daxis))
    phi = (N.sum(axis=Laxis).sum(axis=Daxis))/(N.sum(axis=Laxis).sum(axis=Maxis).sum(axis=Daxis))

    return alpha,phi

def ComputeN(alpha, phi, N ):
    D, M, L = np.shape(N)
    denom = 0.0

    for d in range(D):
        for m in range(M):
            for l in range(L):
                for i in range(6):
                    for j in range(6):
                            denom += alpha[i]*phi[j]*bool((i+1)+(j+1)==sum[d])
                N[d][m][l] = (alpha[l]*phi[m]*bool((l+1)+(m+1)==sum[d]))/denom
                denom=0.0
    return N


#biased table rows differently
probTable= [0.4,0.4,0.05,0.05,0.05,0.05]
probTable= [1./6,1./6,1./6,1./6,1./6,1./6]

probPlayer= [1./6,1./6,1./6,1./6,1./6,1./6]
probPlayer2= [0.4,0.4,0.05,0.05,0.05,0.05]
probPlayer3= [0.05,0.05,0.05,0.05, 0.4,0.4,]

#vector storing the S observations, sum of both dice
sum = []

D = 3000
for i in range(D):
    playerDice = rollBiasedDice(probPlayer)
    tableDice = rollBiasedDice(probTable)
    sum.append(playerDice + tableDice)


Daxis=0
Maxis=1
Laxis=2

N = np.zeros((D,6,6))

d = np.random.uniform(-0.005,0.005,6)
StdProbs= np.array([1./6,1./6,1./6,1./6,1./6,1./6])+d
StdProbs = StdProbs/np.sum(StdProbs)

alpha=StdProbs
phi=StdProbs
D, M, L = np.shape(N)

for q in range(50):

    denom = 0.0

    #compute N
    for d in range(D):
        for m in range(M):
            for l in range(L):
                for i in range(6):
                    for j in range(6):
                        denom += alpha[i] * phi[j] * bool((i + 1) + (j + 1) == sum[d])
                N[d][m][l] = (alpha[l] * phi[m] * bool((l + 1) + (m + 1) == sum[d])) / denom
                denom = 0.0

    #compute alpha and phi
    NsumAllaxis=(N.sum(axis=Laxis).sum(axis=Maxis).sum(axis=Daxis))
    alpha = (N.sum(axis=Maxis).sum(axis=Daxis)) / NsumAllaxis
    phi = (N.sum(axis=Laxis).sum(axis=Daxis)) / NsumAllaxis


    #N = ComputeN(alpha,phi,N)
    #alpha,phi = ComputeAlphaAndPhi(N,alpha,phi)
    print "alpha---->", alpha
    print "phi------>", phi
