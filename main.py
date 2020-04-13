'''
SaMDP: Sam's baic MDP formulation using Value Iteration
Input: worldSize, epsilon, gamma
outputs: valueFunction
--------------------------
worldSize: [int] controling the sidelength of the square grid to be navigated.
    Naviation always starts from (0,0) towards a goal at (worldSize,worldSize).

epsilon: [float] small number, when the smallest change in the value of a
    location between subsequent time steps is smaller, optimization
    terminates.

gamma: [float] discount factor.

valueFunction: [float array] of worldSize x worldSize listing the value of each
    state such that a greedy gradient ascent qualifies as pi*.
--------------------------

migirditch@gmail.com
Python 3.7
'''
# Imports
import sys  # sys epsilon
import numpy as np  # valueFunction representation
import matplotlib.pyplot as plt  # vis tool

# Support functions


def lookAhead(valueFunciton, action, currentState, gamma):
    # init
    valueHolder = valueFunciton
    [rSize, cSize] = valueFunciton.shape
    if rSize == cSize:
        worldSize = rSize
    else:
        sys.exit('Error in SAMDP:lookAhead: rSize!=cSize')

    newStates = admissableMoves(currentState, worldSize)
    prob = transitionProbability(action, currentState, newStates)
    rewards = expectedReward(newStates)
    index = 0
    for state in newStates:
        r = state[0]
        c = state[1]
        valueFunciton[r][c] = prob[index] * \
            (rewards[index] + gamma * valueHolder[r][c])
        index += 1

    return valueFunciton


def transitionProbability(action, state, newStates):
    pos = state + action
    dist = [max(
        sys.float_info.epsilon,
        (point[0]-pos[0])**2 + (point[1]-pos[1]) ** 2)
        for point in newStates]
    prob = np.ones(len(dist))
    prob /= (dist)
    # sum over columns
    return(prob)


def expectedReward(newStates):

    rewards = np.ones(len(newStates)) * -1
    return(rewards)


def admissableMoves(state, worldSize):

    r = int(state[0])
    c = int(state[1])
    rows = [r]
    cols = [c]
    newStates = []
    # rows
    if r > 0:
        rows.append(r - 1)
    if r < worldSize - 1:  # indexed 0:wS-1
        rows.append(r + 1)
    # cols
    if c > 0:
        cols.append(c - 1)
    if c < (worldSize - 1):  # indexed 0:wS-1
        cols.append(c + 1)
    # package for cartesian product
    for newR in rows:
        for newC in cols:
            nextState = [newR, newC]
            newStates.append(nextState)
    newStates = np.array(newStates)
    return(newStates)


def valueIteration(worldSize, epsilon=0.00001, gamma=1.0):

    actionPrimatives = np.array(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0], [-1, -1], [1, 1], [-1, 1], [1, -1]])

    valueFunction = np.zeros([worldSize, worldSize])
    newValueFunction = np.zeros([worldSize, worldSize])

    # init
    convergence = False  # min( dV < epsilon ), force at least 1 iteration.

    while(not convergence):

        # simultanious update
        for row in range(0, worldSize - 1):
            for col in range(0, worldSize - 1):
                for action in actionPrimatives:
                    state = np.array([row, col])
                    newValueFunction = lookAhead(
                        valueFunction, action, state, gamma)
        # close out
        # min bc reward is <= 0
        diff = newValueFunction - valueFunction
        maxDifference = diff.min()
        convergence = maxDifference <= epsilon
        valueFunction = newValueFunction
    return valueFunction

# end valueIteration()


# run it
value = valueIteration(20)

# plot it
plt.imshow(value)
plt.colorbar()
plt.show()
