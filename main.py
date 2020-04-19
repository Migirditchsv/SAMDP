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
import math # floor
import sys  # sys epsilon
import numpy as np  # valueFunction representation
import matplotlib.pyplot as plt  # vis tool
import seaborn as sns

# Support functions
def lookAhead(valueFunciton, obstacleSet,actionPrimatives, currentState):
    
    # Check for square world
    qFunction = []
    [rSize, cSize] = valueFunciton.shape
    if rSize == cSize:
        worldSize = rSize
    else:
        sys.exit('Error in SAMDP:lookAhead: rSize!=cSize')

    newStates = admissableMoves(currentState, obstacleSet, worldSize)
    rewards = expectedReward(newStates)
    newValues = [valueFunciton[index] for index in newStates]
    stateIndex = range(len(newValues))
    
    # Loop over actions
    for action in actionPrimatives:
        prob = transitionProbability(action, currentState, newStates)
        actionOutcomes =  [prob[i] * (rewards[i] + newValues[i]) 
                 for i in stateIndex]
        qFunction.append( sum(actionOutcomes) )
    return qFunction


def transitionProbability(action, state, newStates):
    pos = ( state[0] + action[0] , state[1] + action[1] )
    size = max(1, len(newStates))
    prob = [(1 if state == pos else 1/size) for state in newStates]
    # Normalize prob
    norm = 1.0 / sum(prob)
    prob = [ i * norm for i in prob ]
    return(prob)


def expectedReward(newStates):

    rewards = [-1] * len(newStates)
    return(rewards)


def admissableMoves(state, obstacleSet, worldSize):
    # Needs rewrite for speed and obstacles.
    # Maybe use another dic { (r,c): [ ]}
    r = int(state[0])
    c = int(state[1])
    rows = [r]
    cols = [c]
    newStates = []
    # rows
    if r > 0: rows.append(r - 1)
    if r < worldSize - 1: rows.append(r + 1)
    # cols
    if c > 0: cols.append(c - 1)
    if c < (worldSize - 1): cols.append(c + 1)
    # package for cartesian product
    for newR in rows:
        for newC in cols:
            nextState = (newR, newC)
            if nextState not in obstacleSet:
                newStates.append(nextState)
    return(newStates)


def valueIteration(worldSize, epsilon=0.0001, gamma=1.0):
    center = math.floor(worldSize/2)
    goalPoint = (center,center)
    obstacleSet = { (3,3), (4,4), (3,4), (4,3), (worldSize-1, worldSize-2)}
    actionPrimatives = [ (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                         (-1, -1), (1, 1), (-1, 1), (1, -1)]
    
    # Policy(state) dict int
    #Init to stationary action
    worldIterator = [(i//worldSize, i%worldSize) for i in  range(worldSize**2)]
    policy = { state:(0,0) for state in worldIterator }
    
    # Value function init
    valueFunction = np.zeros([worldSize, worldSize])
    valueFunction[goalPoint] = 1
    holdValueFunction = valueFunction.copy()

    # Init
    convergence = False  # min( dV < epsilon ), force at least 1 iteration.
    iterations = 0
    
    # Hot loop
    while( iterations < 10): #not convergence):
        
        # simultanious update
        for state in worldIterator:

            if state == goalPoint: continue
            # Skip obstacle points
            if state in obstacleSet:
                valueFunction[state] = None
                policy[state] = (0, 0)
                continue
        
            qFunction = lookAhead(holdValueFunction, obstacleSet, 
                                  actionPrimatives, state)
            
            # find max and argmax of q. Lists are w/o built in max :(
            f = lambda i:qFunction[i]
            argMax = max(range(len(qFunction)), key = f )
            actMax = actionPrimatives[argMax]
            print('main::valueIteration actmax: ', actMax)
            print('main::valueIteration qFunction: ', qFunction)

            
            # Update Values
            valueFunction[state] = max( qFunction )
            
            # Update Policy
            policy[state] = actMax

        # close out
        diff = abs( holdValueFunction - valueFunction )
        maxDifference = np.nanmax(diff)
        convergence = maxDifference <= epsilon
        print('v_k+1(s) - v_k(s) = ', maxDifference)
        
        # Update values
        holdValueFunction = valueFunction.copy()    
        iterations += 1
        # Output
        print('Iterations Passed: ', iterations)
        
        # plot it
        # Bounds
        minValue = np.nanmin(valueFunction)
        maxValue = np.nanmax(valueFunction)
        
        figValue, axValue = plt.subplots(figsize=(20,20))
        sns.heatmap(valueFunction, vmin = minValue, vmax=maxValue, annot=True, fmt="0.2f", linewidths=.01, ax=axValue)
        
        figPolicy, axPolicy = plt.subplots(figsize=(20,20))
        X = [ state[0] for state in worldIterator]
        Y = [ state[1] for state in worldIterator]
        U = [ policy[state][0] for state in worldIterator]
        V = [ policy[state][1] for state in worldIterator]
        q = axPolicy.quiver(X,Y,U,V)
        
        plt.pause(0.05)
        plt.show()
    return valueFunction

# end valueIteration()


# run it
value = valueIteration(20)





# %%
