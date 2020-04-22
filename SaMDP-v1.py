#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:27:27 2020

@author: sam
"""
import numpy as np

class SAMDP:
    ''''SAMDP: Sam's MDP readable and editable MDP-VI solver.
    Inputs:
    stateSpace[nparray]: a square occupancy grid with None values for
    inaccessibel points, negative values for transient states and positive
    values for goal states. This allows rewardFunction(s) to be derived.
    
    actionPrimatives[list of tuples]: list of N-tuples where N is the
        dimension of stateSpace. Dynamic vs. kinetic agnostic in SAMDP, but
        fed to transitionModel() so dynamic/kinetic control can be set there.
        
    transitionModel{ ((sx,sy),(ax,ay)):[[s'x,s'y,prob]]} dict mapping a state,
        an action to a normalized list of lists of possible new states & their
        probs.
    '''
    
    def __init__(self, _stateSpace, _actionPrimatives,
                 _transitionModel, _epsilon=0.01 ):
    
        # Safety checks
        #eventually upgrade to handle rectangular 
        [self.rowSize, self.colSize] = valueFunciton.shape
        if rSize == cSize: worldSize = rSize
        else:sys.exit('Error in SAMDP.__init__: rowSize!=colSize')
        
        # Self copies
        self.vFunction = _stateSpace # Init from state space
        self.actionPrimatives = _actionPrimatives
        self.transitionModel = _transitionModel
        self.epsiolon = _epsilon
        
        # Internal Params
        self._worldSize = self.rowSize * self.colSize
        self._stateIterator = [(i//self.colSize, i%self.rowSize)
                          for i in  range(worldSize)]
        _obstacleSet = { state for state in self._stateIterator
                        if self.vFunction[state] != None}
        
        #public Params
        self.policy = {state:self._actionPrimatives[0]
                       for state in self._stateIterator}
        self.vFunction = {state:self.stateSpace[state]
                          for state in self._stateIterator}
        
        
## Driver functions

# stateSpace
def stateSpaceGenerator(worldSize, obstacleFraction):
    if obstacleFraction > 1.0 or obstacleFraction < 0.0:
        sys.exit('Error in SAMDP stateSpaceGenerator: obstacleFraction must be between 0.0 and 1.0')

    center = math.floor(worldSize/2)
    goalPoint = (center,center)
    
    # flat list of states, useful for comprehensions
    worldIterator = [(i//worldSize, i%worldSize) for i in  range(worldSize**2)]
    obstacleSet = set( random.sample(worldIterator,
                                     obstacleFraction * ( worldSize**2 ) ) )
    if goalPoint in obstacleSet: obstacleSet.remove(goalPoint)

    stateSpace = np.zeros((worldSize,worldSize))

    for state in worldIterator:
        if state == goalPoint: stateSpace[state] = 1
        if state in obstacleSet: stateSpace[state] = None
        else: stateSpace[state] = -1
    
    return(stateSpace)

# ActionPrimatives
actionPrimatives = [ (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                         (-1, -1), (1, 1), (-1, 1), (1, -1)]
# Transition model
def transitionModel( stateSpace, actionPrimatives, noiseLevel):
    
    (rowSize, colSize) = stateSpace.shape
    
    for r in rowSize:
        for c in colSize:
            
            state = (r,c)
            
            #Build list of states accessible from state=(r,c) under any action
            rows = [r]
            cols = [c]
            accessibleStates = []
            # rows
            if r > 0: rows.append(r - 1)
            if r < worldSize - 1: rows.append(r + 1)
            # cols
            if c > 0: cols.append(c - 1)
            if c < (worldSize - 1): cols.append(c + 1)
            for newR in rows:
                for newC in cols:
                    nextState = (newR, newC)
                    if stateSpace[nextState] != None:
                        accessibleStates.append(nextState)
            
            for action in actionPrimatives:
                
                # find prob of transition to newState in accessibleStates
                actionPosition = ( state[0] + action[0], state[1] + action[1] )
                prob = [ 1.0 if checkState == actionPosition else noiseLevel
                        for checkState in accessibleStates ]
                norm =  1.0 / sum(prob)
                prob = [ i * norm for i in prob ]
                stateActionUpdatePDF = [ ( newState[i],prob[i] )
                                        for i in range(len(accessibleStates))]
                transition[(state, action)] = stateActionUpdatePDF
    return(transition)
    


## Build it
    
jersy = SAMDP( newJersey, badDriver, shittyRoads,