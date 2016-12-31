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

#dimensions
K=10000
N = 100
#non biased
#probTable= [1./6,1./6,1./6,1./6,1./6,1./6]
#probPlayer= [1./6,1./6,1./6,1./6,1./6,1./6]

#biased towards 5 and 6
#probTable= [1./8,1./8,1./8,1./8,1./4,1./4]
#probPlayer= [1./8,1./8,1./8,1./8,1./4,1./4]

#biased table rows differently
probTable0= [0.4,0.4,0.05,0.05,0.05,0.05]
probTable1= [0.05,0.05,0.05,0.05,0.4,0.4]
probPlayer= [1./6,1./6,1./6,1./6,1./6,1./6]

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


plt.hist(sum,11,(2,12))
plt.show()

