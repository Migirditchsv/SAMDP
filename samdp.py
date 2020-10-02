#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:27:27 2020

@author: sam
"""
from matplotlib.pyplot import figure
from scipy.sparse.linalg import inv  # matrix inevrsion for MFPT
from scipy.sparse import csc_matrix  # matrix inevrsion for MFPT
import numpy as np
import math as m  # floor for terrain generation
import random  # random obstacle generation
import sys  # sys.exit for warnings
import matplotlib.pyplot as plt  # vis tool
import seaborn as sns  # vis tool
import os  # datafile management
import time  # DEBUGGING
# import sandbox  # better matrix inversion


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
    gamma = 0.9  # exponential discount factor
    maxSteps = 300  # Max allowable solver steps
    mfptRefreshPeriod = 5

    # warmup
    # num of zeros to pad frame counter
    frameMagnitude = m.ceil(m.log(maxSteps) / m.log(10))
    folderPath = './RenderingFrames'  # where to dump pre-rendered frames.

    def __init__(self, _stateSpace, _actionPrimatives):

        # Safety checks
        # eventually upgrade to handle rectangular domain
        [self.rowSize, self.colSize] = _stateSpace.shape
        self.worldSize = self.rowSize * self.colSize
        if self.rowSize != self.colSize:
            sys.exit('Error in SAMDP.__init__: rowSize!=colSize')
        # check for and create data file
        if not os.path.exists(self.folderPath):
            os.mkdir(self.folderPath)
            os.mkdir(self.folderPath+'/Default')

        # ffmpeg crashes if it tries to write to an existing file
        # if os.path.exists('out.mp4'):
        #    os.remove('out.mp4')

        # Self copies
        self.valueFunction = _stateSpace.copy()  # Init from state space
        self.holdValueFunction = np.copy(_stateSpace)
        self.rewards = np.copy(_stateSpace)
        self.actionPrimatives = _actionPrimatives

        # Internal Params
        self._stateIterator = [(i//self.colSize, i % self.rowSize)
                               for i in range(self.worldSize)]
        self._actionNorm = 1.0 / len(self.actionPrimatives)
        # value of _stateSpace to look for in id'ing goal pts.
        self.goalValue = np.nanmax(self.valueFunction)
        # value of inaccessible points
        self.obstacleValue = float(np.nanmin(self.valueFunction))
        self.normalValue = float(np.nanmin(abs(self.valueFunction)))

        # public vars
        # Stores mfpt scores by state index
        self.mfpt = []
        self.mfptRanked = []
        self.frameBuffer = 0  # number of frames being stored in ram
        self.frameIndex = 1
        self.solverIterations = 1
        self.updateList = self._stateIterator  # change in loop for partial update
        self.goalSet = {state for state in self._stateIterator
                        if self.valueFunction[state] == self.goalValue}
        self.obstacleSet = {state for state in self._stateIterator
                            if self.valueFunction[state] == self.obstacleValue}
        self.normalSet = {state for state in self._stateIterator
                          if state not in self.obstacleSet.union(self.goalSet)}
        self.problemSet = {state for state in self._stateIterator
                           if state not in self.goalSet}
        self.policy = {state: self.actionPrimatives[0]
                       for state in self._stateIterator}
        self.maxDifference = 9999999999999
        # estimate of convergence condition
        self.convergenceThresholdEstimate = 0.1 * self.normalValue  # (
        # 0.5 * abs(self.goalValue - self.obstacleValue) / self.worldSize)**2

        # Internal data objects
        self._frames = []  # list of plots to save
        self._frameIndex = 0

    # Hot loop functions
    def _admissableMoves(self, currentState):

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
                newStates.append(nextState)

        return(newStates.copy())

    def _transitionProbability(self, action, state, newStates):
        # default transitions, to be depreciated.
        pos = (state[0] + action[0], state[1] + action[1])
        size = max(1, len(newStates) - 1)
        prob = [0.9 if state == pos else (0.1/size) for state in newStates]
        # Normalize prob
        if sum(prob) == 0:
            return(prob)
        norm = 1.0 / sum(prob)
        prob = [i * norm for i in prob]
        return(prob)

    def _lookAhead(self, currentState):
        newStates = self._admissableMoves(currentState)
        rewards = [self.rewards[index] for index in newStates]
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
        return qFunction

    def _fixedPolicyLookAhead(self, currentState):
        # Sets the value of a state to a one step lookahead.
        action = self.policy[currentState]
        newStates = self._admissableMoves(currentState)
        rewards = [self.rewards[index] for index in newStates]
        newValues = [self.holdValueFunction[index] for index in newStates]
        prob = self._transitionProbability(action, currentState, newStates)

        # fast compute state value under fixed action
        value = 0.0
        for stateIndex in range(len(newStates)):
            value += prob[stateIndex] * \
                (rewards[stateIndex] + self.gamma * newValues[stateIndex])
        # Update State Value
        if currentState not in self.obstacleSet:
            self.valueFunction[currentState] = value

    def _mfptLookAhead(self, currentState):
        newStates = self._admissableMoves(currentState)
        rewards = [self.mfpt[index] for index in newStates]
        stateIndex = range(len(newStates))

        qFunction = []

        # Loop over actions
        for action in self.actionPrimatives:
            # prob of landing in each new state after action
            prob = self._transitionProbability(action, currentState, newStates)
            # risk*reward coefficients for each possible outcome of the action
            actionOutcomes = [(prob[i] * rewards[i]) for i in stateIndex]
            qFunction.append(sum(actionOutcomes))

        # Assign action MINIMIZING mfpt score of next step
        # find min and argmin of q.
        def f(i): return qFunction[i]
        argMin = min(range(len(qFunction)), key=f)
        actMin = self.actionPrimatives[argMin]
        # Update Policy
        self.policy[currentState] = actMin

    # Iteration functions

    def policyIterationStep(self):
        # Value convergence step
        count = 0
        while count < self.worldSize:
            for state in self.problemSet:
                # Lookahead
                self._fixedPolicyLookAhead(state)
                count += 1

        # policy update step
        for state in self.problemSet:
            qFunction = self._lookAhead(state)
            # find max and argmax of q.
            def f(i): return qFunction[i]
            argMax = max(range(len(qFunction)), key=f)
            actMax = self.actionPrimatives[argMax]
            # Update Policy
            self.policy[state] = actMax

        # close out
        diff = abs(self.holdValueFunction - self.valueFunction)
        hold = np.nanmax(diff)
        self.maxDifference = hold

        # Update values
        self.holdValueFunction = self.valueFunction.copy()
        self.solverIterations += 1

    def mfptPolicyIteration(self):

        # Define update order if undef
        if len(self.mfptRanked) == 0:
            self.mfptRanked = self.problemSet.copy()
        # First propagate value on all states ordered by current mfpt ranking
        count = 0
        while count < self.worldSize:
            for state in self.mfptRanked:
                # Lookahead
                self._fixedPolicyLookAhead(state)
                count += 1

        # Then do a second sweep updating policies as neighboring values have shifted
        for state in self.mfptRanked:
            qFunction = self._lookAhead(state)
            # find max and argmax of q.
            def f(i): return qFunction[i]
            argMax = max(range(len(qFunction)), key=f)
            actMax = self.actionPrimatives[argMax]
            # Update Policy
            self.policy[state] = actMax

        # Update mfpt ranking if empty or if refresh period
        if self.solverIterations % self.mfptRefreshPeriod == 0:
            self.mfptRank(selectionRatio=1.0)

        # Update policy to maximize MFPT value

        # close out
        diff = abs(self.holdValueFunction - self.valueFunction)
        hold = np.nanmax(diff)
        self.maxDifference = hold

        # Update values
        self.holdValueFunction = self.valueFunction.copy()
        self.solverIterations += 1

    def hybridIterationStep(self):
        # A sumultainous update to value and then policy optimimzation
        for state in self.updateList:

            # Skip goal states
            if state in self.goalSet:
                self.policy[state] = (0, 0)
                continue

            # Lookahead
            qFunction = self._filookAhead(state)

            # find max and argmax of q. Lists are w/o built in max :(
            def f(i): return qFunction[i]
            argMax = max(range(len(qFunction)), key=f)
            actMax = self.actionPrimatives[argMax]

            # Update Policy
            self.policy[state] = actMax

            # Update Values if not obstacle
            if state in self.obstacleSet:
                continue
            self.valueFunction[state] = max(qFunction)

        # close out
        diff = abs(self.holdValueFunction - self.valueFunction)
        hold = np.nanmax(diff)
        self.maxDifference = hold

        # Update values
        self.holdValueFunction = self.valueFunction.copy()
        self.solverIterations += 1
        # print("SAMDP.hybridIterationStep step: ", self.solverIterations,
        #      "max difference: ", self.maxDifference)

    # Rendering methods

    def createDataDir(self, runPath):
        path = self.folderPath+'/'+runPath
        if os.path.exists(path):
            os.system('rm -rf '+path)  # generally unsafe but...
        os.mkdir(path)

    def renderFrame(self, title='', renderValueGradient=False):
        # renders the current policy and value functions into a single frame
        # stored in self._frames[].

        # Settings
        offset = 0.5  # To align vectors with heatmap plot

        # Bounds
        meanVal = np.nanmean(self.valueFunction)
        minValue = meanVal - (0.5 * abs(meanVal))
        maxValue = meanVal + (0.5 * abs(meanVal))

        # Format Data
        X = [state[1] + offset for state in self._stateIterator]
        Y = [state[0] + offset for state in self._stateIterator]
        U = [self.policy[state][1] for state in self._stateIterator]
        V = [-self.policy[state][0] for state in self._stateIterator]

        # plot it
        fig = figure()
        plt.quiver(X, Y, U, V, scale=50, zorder=10, color='cyan')
        # plt.set_ylim(plt.get_ylim())
        sns.heatmap(self.valueFunction, annot=False, cbar=True, fmt="0.2f",
                    linewidths=.01, annot_kws={"size": 4}, zorder=0)

        # title
        string = [title, 'Solver Iteration:'+str(self.solverIterations)]
        string = '\n'.join(string)
        plt.title(string, fontsize=15)

        # housekeeping
        self._frames.append(fig)
        plt.close(fig)
        self.frameBuffer += 1

        if renderValueGradient == True:
            self.renderValueGradient()

    def writeOutFrameBuffer(self, runPath='Default'):
        # the buffer of frames for making movies consumes ram, slowing
        # computation, but writing is also slow. Reccomend writing every ~10
        # frames to avoid crunch
        for figure in self._frames:
            frameLabel = str(self.frameIndex).zfill(self.frameMagnitude)
            fileName = 'img'+frameLabel+'.png'
            filePath = self.folderPath+'/'+runPath+'/'+fileName
            print('saving: ', filePath)
            figure.savefig(filePath, bbox_inches='tight', pad_inches=0)
            self.frameIndex += 1
        # clear written frames
        self._frames = []

    def renderGIF(self, runTitle='Default'):
        # ffmpeg exits if writing to existing file
        if os.path.exists('out.mp4'):
            os.remove('out.mp4')

        # Stitch into gif
        # command = 'ffmpeg -framerate 2 -i '+self.folderPath+'/'+runTitle + \
        #     '/img0%'+str(self.frameMagnitude) + \
        #     'd.png -g 8 -c:v libx264 -r 2 '+runTitle+'.mp4'
        command = 'ffmpeg -framerate 2 -i '+self.folderPath+'/'+runTitle+'/img%0' + \
            str(self.frameMagnitude)+'d.png -g 8 -c:v libx264 -r 2 ' + \
            self.folderPath+'/'+runTitle+'/'+runTitle+'.mp4'
        print('COMMAND: ', command)
        os.system(command)

    def renderValueGradient(self):

        # map states to max derivatives in a dic
        valueGradients = {state: 0.0 for state in self.normalSet}
        for state in self._stateIterator:

            # Init values for state
            stateReward = self.rewards[state]
            stateValue = self.valueFunction[state]
            maxValueGradient = 0
            # find state neighbors
            neighborSet = self._admissableMoves(state)
            for neighbor in neighborSet:
                neighborReward = self.rewards[neighbor]
            # compute reward and value derivative for neighbor
                neighborValue = self.valueFunction[neighbor]
                deltaValue = neighborValue - stateValue
                deltaReward = neighborReward - stateReward
                derivative = abs(deltaValue - deltaReward)
                if derivative > maxValueGradient:
                    maxValueGradient = derivative
            # store top gradient in dic
            valueGradients[state] = maxValueGradient

        # Display gradient
        gradientField = np.empty_like(self.valueFunction)
        # fillin
        for item in valueGradients.items():
            #item[0] is key, item[1] is value
            state = item[0]
            localGradient = item[1]
            gradientField[state] = localGradient
        # Display
        figure, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.heatmap(gradientField, annot=True, cbar=False, fmt="0.2f",
                    linewidths=.01, ax=ax, annot_kws={"size": 14})
        titleString = ' '.join(['Value Gradient at Iteration',
                                str(self.solverIterations)])
        plt.suptitle(titleString, fontsize=35)
        frameLabel = str(self.solverIterations).zfill(self.frameMagnitude)
        fileName = 'gradient'+frameLabel+'.png'
        filePath = self.folderPath+'/'+fileName
        print('saving: ', filePath)
        figure.savefig(filePath, bbox_inches='tight', pad_inches=0)
        plt.close()

    def renderMFPT(self):
        mfptGrid = np.zeros((self.rowSize, self.colSize))
        for state in self._stateIterator:
            stateIndex = self.stateToIndex(state)
            mfptGrid[state] = self.mfpt[stateIndex]
        figure, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.heatmap(mfptGrid, annot=False, cbar=True, fmt=".2g",
                    linewidths=.01, ax=ax, annot_kws={"size": 14})
        titleString = ' '.join(['MFPT Values at Iteration',
                                str(self.solverIterations)])
        # ax.invert_yaxis()
        plt.suptitle(titleString, fontsize=35)
        frameLabel = str(self.solverIterations).zfill(self.frameMagnitude)
        fileName = 'MFPT'+frameLabel+'.png'
        filePath = self.folderPath+'/'+fileName
        print('saving: ', filePath)
        figure.savefig(filePath, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Ranking algs

    def DijkstraValueSeed(self):
        # Runs the dijkstra alg from the first goal state,
        # moving along action primatives.

        # unvisited states
        unvisited = self.normalSet.copy()
        # start from goal states
        currentStates = self.goalSet
        # fringe is empty
        fringe = set()
        # Init values to lowest possible
        dijkstraValue = np.empty_like(self.valueFunction)

        for state in self._stateIterator:
            if state in self.goalSet:
                dijkstraValue[state] = self.goalValue
            else:
                dijkstraValue[state] = self.obstacleValue

        # Loop while unvisited not empty
        while len(unvisited) > 0:

            # add to fringe from successor of current states
            for state in currentStates:
                # Add successors of state to fringe
                successors = self._admissableMoves(state)
                # update successors values and check if new fringe
                for newState in successors:
                    if newState in self.obstacleSet:
                        continue
                    if newState in self.goalSet:
                        continue
                    newValue = self.rewards[newState] + dijkstraValue[state]
                    if newValue > dijkstraValue[newState]:
                        dijkstraValue[newState] = newValue
                    if newState in unvisited:
                        unvisited.remove(newState)
                        fringe.add(newState)
            currentStates = fringe.copy()
            fringe.clear()

        # Push dijkstra values into value function
        self.valueFunction = dijkstraValue.copy()
        self.holdValueFunction = dijkstraValue.copy()

        return

    def gradientRank(self, selectionRatio):
        # check gradient for all states. Really is comparing max slope in value
        # minus that direction's reward for non-obstacle neighbors

        # set rank size
        selectionSize = m.floor(len(self._stateIterator) * selectionRatio)
        gradientRank = []

        # map states to max derivatives in a dic
        valueGradients = {state: 0.0 for state in self.normalSet}
        for state in self._stateIterator:

            # Init values for state
            stateReward = self.rewards[state]
            stateValue = self.valueFunction[state]
            maxValueGradient = 0
            # find state neighbors
            neighborSet = self._admissableMoves(state)
            for neighbor in neighborSet:
                neighborReward = self.rewards[neighbor]
            # compute reward and value derivative for neighbor
                neighborValue = self.valueFunction[neighbor]
                deltaValue = neighborValue - stateValue
                deltaReward = neighborReward - stateReward
                derivative = abs(deltaValue - deltaReward)
                if derivative > maxValueGradient:
                    maxValueGradient = derivative
            # store top gradient in dic
            valueGradients[state] = maxValueGradient

        # Rank
        for index in range(selectionSize):
            state = max(valueGradients, key=valueGradients.get)
            gradientRank.append(state)
            valueGradients.pop(state)

        return gradientRank.copy()

    def mfptRank(self, selectionRatio, displayMFPT=False):
        # Init

        updateNumber = m.floor(selectionRatio * self.worldSize)
        builderMtx = np.zeros((self.worldSize, self.worldSize))
        negIdentityVec = np.full((self.worldSize, 1), -1)

        # Populate transition probability matrix

        for state in self._stateIterator:
            stateIndex = self.stateToIndex(state)
            action = self.policy[state]
            neighbors = self._admissableMoves(state)
            probs = self._transitionProbability(action, state, neighbors)
            # convert neighbor states to state indicies and insert probs
            # remember -1 on diagonal.
            probIndex = 0
            for neighbor in neighbors:
                neighborIndex = self.stateToIndex(neighbor)
                builderMtx[stateIndex,
                           neighborIndex] = probs[probIndex] - (stateIndex == neighborIndex)
                probIndex += 1

        # tic = time.time()
        # # Do efficient sparse matrix inversion
        # transProbMtx = csc_matrix(builderMtx)
        # invTransProb = inv(transProbMtx)
        # invTransProb = invTransProb.A  # cast to np array
        # result = np.dot(invTransProb, negIdentityVec)
        # self.mfpt = [result[i][0] for i in range(len(result))]
        # toc = time.time()
        # print('scipy based inversion time: ', toc-tic)

        # Numpy Version
        invTransProb = np.linalg.inv(builderMtx)
        self.mfpt = invTransProb @ negIdentityVec

        mfptRanked = sorted(range(len(self.mfpt)), key=self.mfpt.__getitem__)

        self.mfptRanked = [self.indexToState(mfptRanked[i])
                           for i in range(0, updateNumber)]


# Support functions


    def indexToState(self, stateIndex):
        return((stateIndex//self.rowSize, stateIndex % self.colSize))

    def stateToIndex(self, state):
        return(state[0]*self.colSize + state[1])

    def scoreWalk(self, initialState):
        steps = 0
        cost = 0.0
        state = initialState

        while (state not in self.goalSet and steps < 2 * self.worldSize):

            # Probabylistically transition
            action = self.policy[state]
            admissableStates = self._admissableMoves(state)
            transitionProbs = self._transitionProbability(action,
                                                          state,
                                                          admissableStates)
            # Uncomment for random transitions
            state = random.choices(admissableStates, transitionProbs, k=1)
            state = state[0]
            if state in self.goalSet:
                continue

            # Uncomment for deterministic transitions
            #state = tuple([sum(x) for x in zip(state, action)])
            # if state not in admissableStates:
            #    state = admissableStates[0]
            # tally costs
            cost += self.rewards[state]
            steps += 1
        return(cost)


# Benchmark Functions


    def averageCost(self, prevPolicy, trials, samplePeriod):
        avgCost = 0.0
        deltaPolicy = 0  # number of changed states between policy iterations
        for state in self._stateIterator:

            # Check delta policy
            if self.policy[state] != prevPolicy[state]:
                deltaPolicy += 1

            if ((state[0]+state[1]) % samplePeriod) == 0:
                for i in range(trials):
                    (cost) = self.scoreWalk(state)
                    avgCost += cost

        # normalize
        avgCost /= self.worldSize * trials
        deltaPolicy /= self.worldSize
        return [deltaPolicy, avgCost]

    def evolvingWalkInit(population, initStates):
        position = np.zeros((population, 1))
        cost = np.zeros(dtype=float)


# Setup functions

# stateSpace


def stateSpaceGenerator(worldSize, obstacleFraction=0.0):

    # protections
    if obstacleFraction > 1.0 or obstacleFraction < 0.0:
        sys.exit(
            'Error in samdp.py stateSpaceGenerator: obstacleFraction must be between 0.0 and 1.0')

    # the object
    stateSpace = np.zeros((worldSize, worldSize))

    # parameters
    goalValue = 1.0
    obstacleValue = -1.0  # -0.5 * (goalValue / worldSize**2)
    passiveValue = obstacleValue / worldSize
    center = m.floor(worldSize/2.0)
    quarter = m.floor(center / 2.0)

    # Init
    goalSet = {(1, worldSize-2)}
    obstacleSet = set()

    # flat list of states, useful for comprehensions
    stateIterator = [(i//worldSize, i % worldSize)
                     for i in range(worldSize**2)]
    obstacleNumber = int(obstacleFraction * (worldSize**2))

    # Random obstacle
    # obstacleSet = set(random.sample(stateIterator, k=obstacleNumber))

    # cross in the middle
    for state in stateIterator:
        BRow1 = (state[0] > center - quarter) and state[1] == center
        BRow2 = (state[0] < center + quarter) and state[1] == center
        if (BRow1 and BRow2):
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

    for state in stateIterator:
        # soft obstacle on top has 0.5*obstacle cost
        feature1 = (state[0] <= center - quarter) and state[1] == center
        # right of center horizontal hole
        feature2 = (state[1] > center and
                    state[1] != center + quarter and
                    state[0] == center)
        # bottom right checkerboard
        feature3 = (state[0] > center) and (state[1] > center) and \
            (((state[0]+state[1]) % 2) == 0)  # checkerboard pattern

        if state in obstacleSet:
            stateSpace[state] = obstacleValue
        elif state in goalSet:
            stateSpace[state] = goalValue
        elif (feature1):
            stateSpace[state] = 0.5 * obstacleValue
        elif (feature2):
            obstacleSet.add(state)
            stateSpace[state] = obstacleValue
        elif (feature3):
            stateSpace[state] = 0.5*obstacleValue
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
