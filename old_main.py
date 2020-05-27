#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import os # frameRender folder check
import math as m # floor, ceil and log for gif indexing
import random # random.sample for obstacle generation
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
    for newR in rows:
        for newC in cols:
            nextState = (newR, newC)
            if nextState not in obstacleSet:
                newStates.append(nextState)
    return(newStates)


def valueIteration(worldSize, epsilon=0.01, gamma=0.9):
    center = m.floor(worldSize/2)
    goalPoint = (0,worldSize-1) #(center,center)
    
    # flat list of states, useful for comprehensions
    worldIterator = [(i//worldSize, i%worldSize) for i in  range(worldSize**2)]
    obstacleSet = { (0,i) for i in range(1,worldSize-3)}#set( random.sample(worldIterator, 5 * worldSize) )
    if goalPoint in obstacleSet: obstacleSet.remove(goalPoint)
    actionPrimatives = [ (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                         (-1, -1), (1, 1), (-1, 1), (1, -1)]
    
    # Policy(state) dict int
    #Init to stationary action
    policy = { state:(0,0) for state in worldIterator }
    
    # Value function init
    valueFunction = np.zeros([worldSize, worldSize])
    valueFunction[goalPoint] = 1
    holdValueFunction = valueFunction.copy()

    # Init
    convergence = False  # min( dV < epsilon ), force at least 1 iteration.
    iterations = 0
    # fig,(axPolicy, axValue) = plt.subplots(nrows=1,ncols=2,
    #                                        figsize=(50,30))
    frames = [] # Frames for gif output
    
    # Hot loop
    while(not convergence):
        
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
            
            # Update Values
            valueFunction[state] = gamma * max( qFunction )
            
            # Update Policy
            policy[state] = actMax

        # close out
        diff = abs( holdValueFunction - valueFunction )
        maxDifference = np.nanmax(diff)
        convergence = maxDifference <= epsilon
        
        # Update values
        holdValueFunction = valueFunction.copy()    
        iterations += 1
        # Output
        print('Iterations Passed: ', iterations, 'delta max: ', maxDifference)
        
        # plot it
        fig,(axPolicy, axValue) = plt.subplots(nrows=1,ncols=2,
                                           figsize=(50,30))
        # Bounds
        minValue = np.nanmin(valueFunction)
        maxValue = np.nanmax(valueFunction)
        
        # axPolicy.cla()
        # axValue.clear() # labels are counted as axes and need to be removed
        
        # (row, col) coordinate form in world iterator
        X = [ state[1] for state in worldIterator]
        Y = [ state[0] for state in worldIterator]
        U = [ policy[state][1] for state in worldIterator]
        V = [ -policy[state][0] for state in worldIterator]
        #U.reverse()
        #V.reverse()
        q = axPolicy.quiver(X,Y,U,V, scale=25)
        axPolicy.set_ylim(axPolicy.get_ylim()[::-1]) 
        v = sns.heatmap(valueFunction, vmin = minValue, vmax=maxValue,
                        annot=True, cbar=False, fmt="0.2f", linewidths=.01,
                        ax=axValue, annot_kws={"size": 4})
        
        #fig.canvas.draw()
        frames.append(fig)#([axPolicy, axValue,])
        plt.close(fig)
        
    #close out
    # check for and create file
    folderPath = './RenderingFrames'
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    else:
        command = 'rm -rf '+folderPath
        os.system(command)
        os.mkdir(folderPath)
        
        # ffmpeg crashes if it tries to write to an existing file
    if os.path.exists('out.mp4'):
        os.remove('out.mp4')
        
    #order of mag. frame index 0 padding
    maxFrames = m.ceil( m.log(len(frames)) / m.log(10) )
    frameIndex = 1
    for figure in frames:
        frameLabel = str(frameIndex).zfill(maxFrames)
        fileName = 'img'+frameLabel+'.png'
        filePath = folderPath+'/'+fileName
        print('saving: ', filePath)
        figure.savefig(filePath, bbox_inches = 'tight', pad_inches = 0) 
        frameIndex += 1

    return valueFunction, maxFrames, folderPath

# end valueIteration()

def gifStitch(maxFrames,folderPath):

    # Stitch into gif
    command = 'ffmpeg -framerate 2 -i '+folderPath+'/img%0'+str(maxFrames)+'d.png -c:v libx264 -r 2 out.mp4'
    os.system( command )

# run it
value, maxFrames, folderPath = valueIteration(20, epsilon= 0.01, gamma = 0.5)
del value
gifStitch(maxFrames, folderPath)




# %%
