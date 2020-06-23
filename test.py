#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""
import time # testing and comparison
import samdp
import os  # play gif at end

# Controls
# side length of square state space
environmentSize = 20
# pre-seed value function?
dijkstraSeed =  True
# run mfpt analysis every X steps. It is expensive.
mfptRefreshPeriod = 99999
# top X percent of mfpt scores to put in update list.
mfptUpdateRatio = 0.3
# Number of random starting states to compute each mfpt score from.
mfptRolloutNumber = 30
# run a global sweep every X steps.
globalSweepPeriod = 1

# Initialize Environment
stateSpace = samdp.stateSpaceGenerator(environmentSize)

# When defining action primatives, the iteration schemes break ties between
# expected utility of actions by the order they appear in the primatives list.
# null actions (0,0), should therefore always be listed LAST
directional8 = [ (1, 0), (-1, 0), (0, 1), (0, -1),
                (-1, -1), (1, 1), (-1, 1), (1, -1),(0, 0)]

directional4 = [ (1, 0), (-1, 0), (0, 1), (0, -1),(0, 0)]

actionPrimatives = directional8

transitionModel = samdp.transitionModelGenerator(
    stateSpace, actionPrimatives, 0.2)

demo = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
demo.renderFrame()

print('Test initalized\n')
# pre-compute
if dijkstraSeed == True:
    dijkstraStart = time.time()
    demo.DijkstraValueSeed()
    dijkstraStop = time.time()
    print('Dijkstra Run Time: ', dijkstraStop - dijkstraStart)


demo.updateList = demo.normalSet
demo.policyIterationStep()
demo.renderFrame()
mfptStart = time.time()
mfptUpdateList = demo.mfptRank(mfptUpdateRatio, mfptRolloutNumber)
mfptStop = time.time()
print('MFPT Run Time: ', mfptStop - mfptStart)

# ???read shoubik first!!! demo.policyUpdate() instead of hybrid step

print('pre-processing complete\n')

# Solve
totalTime = 0.0
unconverged = 1
while unconverged:
   
    # update
    print('test.py: step num:', demo.solverIterations)
    print('test.py: delta value: ', demo.maxDifference, '/',demo.convergenceThresholdEstimate)

    # Clock in
    startTime = time.time()
    
    # mfpt usage
    if demo.solverIterations % mfptRefreshPeriod == 0:
        print('MFPT RE-RANK: INIT')
        mfptUpdateList = demo.mfptRank(mfptUpdateRatio, mfptRolloutNumber)
        print('MFPT RE-RANK: COMPLETE')

    if demo.solverIterations % globalSweepPeriod == 0:
        demo.updateList = demo.problemSet
        demo.hybridIterationStep()
        # Convergence can only be tested for after a global update.
        unconverged = demo.maxDifference > demo.convergenceThresholdEstimate
    else:
        demo.updateList = mfptUpdateList
        demo.hybridIterationStep()
    
    # Clock out
    endTime = time.time()
    deltaTime = endTime - startTime
    print('test.py: deltaTime: ',deltaTime)
    totalTime += deltaTime


    # Save conditions
    stepNum = demo.solverIterations
    print('RENDER FRAME CHECK: INIT')
    if stepNum < 20:
        demo.renderFrame()
    elif stepNum > 20 and stepNum % 3 == 0:
        demo.renderFrame()
    elif stepNum < 30 and stepNum % 1 == 0:
        demo.renderFrame()
    print('RENDER FRAME CHECK: COMPLETE')

    if demo.frameBuffer % 60 == 0:
        # One unwritten frame wastes ~10mb of ram, but writing is slow
        print('BUFFER WRITE: INIT')
        demo.writeOutFrameBuffer()
        print('BUFFER WRITE: COMPLETE')
    # iteration finished. New block
    print('\n')
    
# Complete. Report
print('COMPLETE')
print('run time: ',  totalTime,'\n')

# final frame buffer flush
demo.writeOutFrameBuffer()
# stitch to gif
# demo.renderGIF()

# Report Results
#os.system('xdg-open out.mp4')
