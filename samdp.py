#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:27:27 2020

@author: sam
"""
import numpy as np
import math as m  # floor for terrain generation
import random  # random obstacle generation
import sys  # sys.exit for warnings
import matplotlib.pyplot as plt  # vis tool
import seaborn as sns  # vis tool
import os  # datafile management


class SAMDP:
    ''''SAMDP: Sam's MDP readable and editable MDP-VI solver.
    Inputs:
    stateSpace[nparray]: a square occupancy grid with np.nan values for
    inaccessibel points, negative values for transient states and positive
    values for goal states. This allows rewardFunction(s) to be derived.

    actionPrimatives[list of tuples]: list of N-tuples where N is the
        dimension of stateSpace. Dynamic vs. kinetic agnostic in SAMDP, but
        fed to transitionModel() so dynamic/kinetic control can be set there.

    transitionModel{ ((sx,sy),(ax,ay)):[[s'x,s'y,prob]]} dict mapping a state,
        an action to a normalized list of lists of possible new states & their
        probs.
    '''

    # Universal Control Parameters
    #goalValue = 1.0  # value of _stateSpace to look for in id'ing goal pts.
    #obstacleValue = -0.1  # value of inaccessible points
    #passiveValue = 0.0 # value of transitioning to a normal space
    gamma = 1.0  # exponential discount factor
    maxSteps = 1000  # Max allowable solver steps
    # num of zeros to pad frame counter
    frameMagnitude = m.ceil(m.log(maxSteps) / m.log(10))
    folderPath = './RenderingFrames'  # where to dump pre-rendered frames. Not safe rn

    def __init__(self, _stateSpace, _actionPrimatives,
                 _transitionModel, _epsilon=0.01):

        # Safety checks
        # eventually upgrade to handle rectangular
        [self.rowSize, self.colSize] = _stateSpace.shape
        if self.rowSize != self.colSize:
            sys.exit('Error in SAMDP.__init__: rowSize!=colSize')
        # check for and create data file
        if not os.path.exists(self.folderPath):
            os.mkdir(self.folderPath)
        else:
            command = 'rm -rf '+self.folderPath
            os.system(command)
            os.mkdir(self.folderPath)

            # ffmpeg crashes if it tries to write to an existing file
        if os.path.exists('out.mp4'):
            os.remove('out.mp4')

        # Self copies
        self.valueFunction = _stateSpace  # Init from state space
        self.holdValueFunction = np.copy(_stateSpace)
        self.rewards = np.copy(_stateSpace)
        self.actionPrimatives = _actionPrimatives
        self.transitionModel = _transitionModel
        self.epsilon = _epsilon

        # Internal Params
        self._worldSize = self.rowSize * self.colSize
        self._stateIterator = [(i//self.colSize, i % self.rowSize)
                               for i in range(self._worldSize)]
        self._actionNorm = 1.0 / len(self.actionPrimatives)
         # value of _stateSpace to look for in id'ing goal pts.
        self.goalValue = np.nanmax(self.valueFunction )
         # value of inaccessible points
        self.obstacleValue = np.nanmin(self.valueFunction)

        # public structs
        self.frameBuffer = 0  # number of frames being stored in ram
        self.frameIndex = 1
        self.solverIterations = 0
        self.goalSet = {state for state in self._stateIterator
                        if self.valueFunction[state] == self.goalValue}
        self.obstacleSet = {state for state in self._stateIterator
                            if self.valueFunction[state] == self.obstacleValue}
        self.policy = {state: self.actionPrimatives[0]
                       for state in self._stateIterator}
        self.maxDifference = 9999999999999

        # Internal data objects
        self._frames = []  # list of plots to save
        self._frameIndex = 0

    def hybridIterationStep(self):
        # A sumultainous update to value and then policy optimimzation
        for state in self._stateIterator:

            # Skip goal states
            if state in self.goalSet:
                self.policy[state] = (0,0)
                continue
            # Skip obstacle points
            if state in self.obstacleSet:
                self.policy[state] = (0,0)
                continue
            qFunction = self._lookAhead(state)

            # find max and argmax of q. Lists are w/o built in max :(
            def f(i): return qFunction[i]
            argMax = max(range(len(qFunction)), key=f)
            actMax = self.actionPrimatives[argMax]

            # Update Values
            self.valueFunction[state] =  max(qFunction)

            # Update Policy
            self.policy[state] = actMax

        # close out
        diff = abs(self.holdValueFunction - self.valueFunction)
        hold = np.nanmax(diff)
        self.maxDifference = hold

        # Update values
        self.holdValueFunction = self.valueFunction.copy()
        self.solverIterations += 1
        print("SAMDP.hybridIterationStep step: ", self.solverIterations,
              "max difference: ", self.maxDifference)

    def _lookAhead(self, currentState):
        newStates = self._admissableMoves(currentState)
        rewards = [ self.rewards[currentState] ]* len(newStates)
        newValues = [self.holdValueFunction[index] for index in newStates]
        stateIndex = range(len(newValues))

        qFunction = []

        # Loop over actions
        for action in self.actionPrimatives:
            # prob of landing in each new state after action
            prob = self._transitionProbability(action, currentState, newStates)
            # risk*reward coefficients for each possible outcome of the action
            actionOutcomes = [prob[i] * (rewards[i] + self.gamma * newValues[i])
                              for i in stateIndex]
            qFunction.append(sum(actionOutcomes))
        # qFunction is expectation value for each outcome. Must norm by
        #   #terms = #newStates
        norm = [1.0 / len(newStates)] * len(newStates)
        qFunction = [qFunction[i] * norm[i] for i in range(len(self.actionPrimatives))]
        return qFunction

    def _admissableMoves(self, currentState):
        # Needs rewrite for speed and obstacles.
        # Maybe use another dic { (r,c): [ ]}
        r = int(currentState[0])
        c = int(currentState[1])
        rows = [r]
        cols = [c]
        newStates = []
        # rows
        if r > 0:
            rows.append(r - 1)
        if r < self.rowSize - 1:
            rows.append(r + 1)
        # cols
        if c > 0:
            cols.append(c - 1)
        if c < (self.colSize - 1):
            cols.append(c + 1)
        for newR in rows:
            for newC in cols:
                nextState = (newR, newC)
                # if nextState not in self.obstacleSet:
                newStates.append(nextState)
        return(newStates)

    def _transitionProbability(self, action, state, newStates):
        # default transitions, to be depreciated.
        pos = (state[0] + action[0], state[1] + action[1])
        size = max(1, len(newStates) - 1)
        prob = [(1.0 if state == pos else 0.0/size) for state in newStates]
        # Normalize prob
        if sum(prob) == 0:
            return(prob)
        norm = 1.0 / sum(prob)
        prob = [i * norm for i in prob]
        return(prob)

    def renderFrame(self):
        # renders the current policy and value functions into a single frame
        # stored in self._frames[]
        # plot it
        fig, (axPolicy, axValue) = plt.subplots(nrows=1, ncols=2,
                                                figsize=(50, 30))
        # Bounds
        #meanVal = np.nanmean(self.valueFunction)
        #minValue = meanVal - ( 0.5 * abs(meanVal) )
        #maxValue = meanVal + ( 0.5 * abs(meanVal) )

        # (row, col) coordinate form in world iterator
        X = [state[1] for state in self._stateIterator]
        Y = [state[0] for state in self._stateIterator]
        U = [self.policy[state][1] for state in self._stateIterator]
        V = [-self.policy[state][0] for state in self._stateIterator]
        axPolicy.quiver(X, Y, U, V, scale=25)
        axPolicy.set_ylim(axPolicy.get_ylim()[::-1])
        sns.heatmap(self.valueFunction,
                        #vmin=minValue, vmax=maxValue,
                        annot=True, cbar=False, fmt="0.2f", linewidths=.01,
                        ax=axValue, annot_kws={"size": 20})

        # fig.canvas.draw()
        self._frames.append(fig)  # ([axPolicy, axValue,])
        plt.close(fig)
        self.frameBuffer += 1

    def writeOutFrameBuffer(self):
        # the buffer of frames for making movies consumes ram, slowing
        # computation, but writing is also slow. Reccomend writing every ~10
        # frames to avoid crunch
        for figure in self._frames:
            frameLabel = str(self.frameIndex).zfill(self.frameMagnitude)
            fileName = 'img'+frameLabel+'.png'
            filePath = self.folderPath+'/'+fileName
            print('saving: ', filePath)
            figure.savefig(filePath, bbox_inches='tight', pad_inches=0)
            self.frameIndex += 1
        # clear written frames
        self._frames = []

    def renderGIF(self):
        # extremely ram heavy, delete everything possible
        del self.valueFunction, self.holdValueFunction, self._frames
        # ffmpeg exits if writing to existing file
        if os.path.exists('out.mp4'):
            os.remove('out.mp4')

        # Stitch into gif
        command = 'ffmpeg -framerate 2 -i '+self.folderPath+'/img%0' + \
            str(self.frameMagnitude)+'d.png -g 8 -c:v libx264 -r 2 out.mp4'
        os.system(command)

# Driver functions

# stateSpace


def stateSpaceGenerator(worldSize, obstacleFraction):
    
    # parameters
    goalValue = 10.0
    obstacleValue = -(goalValue / worldSize**2)
    passiveValue = -0.1 * obstacleValue
    
    if obstacleFraction > 1.0 or obstacleFraction < 0.0:
        sys.exit(
            'Error in SAMDP stateSpaceGenerator: obstacleFraction must be between 0.0 and 1.0')

    center = m.floor(worldSize/2.0)
    
    # Init
    goalSet = {(0, worldSize-1), (0,worldSize-2), (1,worldSize-2), (1,worldSize-1)}
    obstacleSet = set()
    
    # flat list of states, useful for comprehensions
    stateIterator = [(i//worldSize, i % worldSize)
                     for i in range(worldSize**2)]
    obstacleNumber = int(obstacleFraction * (worldSize**2))
    
    # Random obstacle
    #obstacleSet = set(random.sample(stateIterator, k=obstacleNumber))

    # cross in the middle
    quarter = m.floor(center / 2.0)
    for state in stateIterator:
        BRow1 = ( state[0] > center - quarter ) and state[1] == center
        BRow2 = ( state[0] < center + quarter ) and state[1] == center
        BCol1 = ( state[1] > center - quarter ) and state[0] == center
        BCol2 = ( state[1] < center + quarter ) and state[0] == center
        if (BRow1 and BRow2) or (BCol1 and BCol2):
            obstacleSet.add(state)
        

    # Walls along border
    for state in stateIterator:
        condition1 = state[0] == 0 or state[0] == worldSize-1
        condition2 = state[1] == 0 or state[1] == worldSize-1
        if condition1 or condition2:
            obstacleSet.add(state)

    # No goals in obstacles, goal status gets prority
    for goal in goalSet:
        if goal in obstacleSet:
            obstacleSet.remove(goal)

    stateSpace = np.zeros((worldSize, worldSize))

    for state in stateIterator:
        if state in obstacleSet:
            stateSpace[state] = obstacleValue
        elif state in goalSet:
            stateSpace[state] = goalValue
        else:
            stateSpace[state] = passiveValue

    return(stateSpace)


# Transition model
def transitionModelGenerator(stateSpace, actionPrimatives, noiseLevel):

    transition = {}

    (rowSize, colSize) = stateSpace.shape
    worldSize = rowSize * colSize

    for r in range(rowSize):
        for c in range(colSize):

            state = (r, c)

            # Build list of states accessible from state=(r,c) under any action
            rows = [r]
            cols = [c]
            accessibleStates = []
            # rows
            if r > 0:
                rows.append(r - 1)
            if r < (rowSize - 2):
                rows.append(r + 1)
            # cols
            if c > 0:
                cols.append(c - 1)
            if c < (colSize - 2):
                cols.append(c + 1)
            for newR in rows:
                for newC in cols:
                    nextState = (newR, newC)
                    if stateSpace[nextState] != np.nan:
                        accessibleStates.append(nextState)

            for action in actionPrimatives:

                # find prob of transition to newState in accessibleStates
                actionPosition = (state[0] + action[0], state[1] + action[1])
                prob = [1.0 if checkState == actionPosition else noiseLevel
                        for checkState in accessibleStates]
                norm = 1.0 / sum(prob)
                prob = [i * norm for i in prob]
                stateActionUpdatePDF = [(accessibleStates[i], prob[i])
                                        for i in range(len(accessibleStates))]
                transition[(state, action)] = stateActionUpdatePDF
    return(transition)


# Driver Script
# stateSpace = stateSpaceGenerator(10, 0.3)

# actionPrimatives = [ (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
#                          (-1, -1), (1, 1), (-1, 1), (1, -1)]

# transitionModel = transitionModelGenerator(stateSpace, actionPrimatives, 0.2)

# demo = SAMDP(stateSpace, actionPrimatives, transitionModel)
