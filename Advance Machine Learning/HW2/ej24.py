import numpy as np
import matplotlib.pyplot as plt
import random as rdm
from scipy.stats import norm
from scipy.stats import bernoulli


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

def ComputeAlpha(Sums, Pi, A, PlayerDice, TableA0Dice, TableA1Dice):

    #compute alpha values for initial observation
    K = len(Sums)
    alpha = np.zeros((K, 2))
    DiceOutcomes = 6

    for j in range(DiceOutcomes):
        for m in range(DiceOutcomes):
            if ((j+1)+(m+1)) == Sums[0]:
                alpha[0,0] += PlayerDice[j] * TableA0Dice[m]
                alpha[0,1] += PlayerDice[j] * TableA1Dice[m]
    alpha[0,0] *= Pi[0]
    alpha[0,1] *= Pi[1]

    #compute alphas recursively
    for k in range(1, K):
        for j in range(DiceOutcomes):
            for m in range(DiceOutcomes):
                if ((j+1)+(m+1)) == Sums[k]:
                    alpha[k, 0] += PlayerDice[j] * TableA0Dice[m]
                    alpha[k, 1] += PlayerDice[j] * TableA1Dice[m]
        alpha[k,0] *= (A[0,0] * alpha[k-1,0] + A[1,0] * alpha[k-1,1])
        alpha[k,1] *= (A[0,1] * alpha[k-1,0] + A[1,1] * alpha[k-1,1])

    return alpha

def ComputeSampling(alpha, A, Sums):
    K = len(Sums)
    R = np.zeros(K, dtype=np.int)  # Sequence sample (empty)

    # Sampling K before going backwards
    p = alpha[K-1,1]/np.sum(alpha[K-1,:])
    R[K-1] = int(bernoulli.rvs(p))

    # Sampling backwards using R recursively
    for k in range(1,K)[::-1]:
        q = A[0, R[k]] * alpha[k-1,0]
        p = A[1, R[k]] * alpha[k-1,1]
        p = p / (q + p)
        R[k-1] = int(bernoulli.rvs(p))
    return R

#----------------------------------------------------------
#dimensions
K=10
N = 1
#biased table rows differently
probTable0= [0.4,0.4,0.05,0.05,0.05,0.05]
probTable1= [0.05,0.05,0.05,0.05,0.4,0.4]

#probTable0 = [1,0,0,0,0,0]
#probTable1 = [1,0,0,0,0,0]

probPlayer= [1./6,1./6,1./6,1./6,1./6,1./6]
#probPlayer = [1,0,0,0,0,0]


#tables is a matrix storing the probabilities of getting a {1,2,3,4,5,6} with its dice
table = np.zeros((2,K), dtype=np.ndarray)
table.fill(probTable0)
print
for i in range(np.shape(table)[1]):
    table[1][i]= probTable1

#players is a vector storing the probabilities of getting a {1,2,3,4,5,6} with her dice
players = np.zeros(N, dtype=np.ndarray)
players.fill(probPlayer)

#vector storing the S observations, sum of both dice
sum = []

for player in players:
    #calculo en que mesa empieza
    nextTable = rdm.randint(0,1)
    for i in range(K):
        playerDice = rollBiasedDice(player)
        tableDice = rollBiasedDice(table[nextTable][i])
        sum.append(playerDice + tableDice)
        #compute in which of the two tables the player will be next time
        ComputeNextTable(nextTable)

print np.shape(sum)
##------------------------------------------------------------------------------------------------
#transition matrix
A = np.array([[1./4,3./4],[3./4,1./4]])
#initial state matrix
Pi = np.array([0.5, 0.5])
#forward algorithm alpha matrix for an observation sequence

alpha = ComputeAlpha(sum, Pi, A, probPlayer, probTable0, probTable1)
Seq = ComputeSampling(alpha, A, sum)


print sum
print Seq