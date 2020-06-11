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
# run mfpt analysis every X steps. It is expensive.
mfptRefreshPeriod = 500
# top X percent of mfpt scores to put in update list.
mfptUpdateRatio = 0.5
# Number of random starting states to compute each mfpt score from.
mfptRolloutNumber = 10
# run a global sweep every X steps.
globalSweepPeriod = 1

# Initialize Environment
stateSpace = samdp.stateSpaceGenerator(20)

directional8 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (-1, -1), (1, 1), (-1, 1), (1, -1)]

directional4 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

actionPrimatives = directional4

transitionModel = samdp.transitionModelGenerator(
    stateSpace, actionPrimatives, 0.2)

demo = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
demo.renderFrame()

print('Test initalized\n')
startTime = time.time()

# pre-compute
dijkjstraStart = time.time()
demo.DijkstraValueSeed()
dijkstraStop = time.time()
print('Dijkstra Run Time: ', dijkstraStop - dijkstraStop)

demo.updateList = demo.normalSet
demo.policyIterationStep()
demo.renderFrame()
mfptStart = time.time()
#demo.mfpt(mfptUpdateRatio, mfptRolloutNumber)
mfptStop = time.time()
print('MFPT Run Time: ', mfptStop - mfptStart)

# ???read shoubik first!!! demo.policyUpdate() instead of hybrid step

print('pre-processing complete\n')

# Solve
while demo.maxDifference > demo.convergenceThresholdEstimate:
   
    # update
    print('test.py: step num:', demo.solverIterations)
    print('test.py: delta value: ', demo.maxDifference, '/',demo.convergenceThresholdEstimate)

    # mfpt usage
    if demo.solverIterations + 1 % mfptRefreshPeriod == 0:
        print('MFPT RE-RANK: INIT')
        mfptUpdateList = demo.mfpt(mfptUpdateRatio, mfptRolloutNumber)
        print('MFPT RE-RANK: COMPLETE')

    if demo.solverIterations % globalSweepPeriod == 0:
        demo.updateList = demo.normalSet
    else:
        demo.updateList = mfptUpdateList

    # update
    print('HYBRID ITERATION STEP: INIT')
    demo.hybridIterationStep()
    print('HYBRID ITERATION STEP: COMPLETE')
    stepNum = demo.solverIterations

    # Save conditions
    print('RENDER FRAME CHECK: INIT')
    if stepNum < 20:
        demo.renderFrame()
    elif stepNum > 20 and stepNum % 3 == 0:
        demo.renderFrame()
    elif stepNum < 30 and stepNum % 4 == 0:
        demo.renderFrame()
    print('RENDER FRAME CHECK: COMPLETE')

    if demo.frameBuffer % 37 == 0:
        # One unwritten frame wastes ~10mb of ram, but writing is slow
        print('BUFFER WRITE: INIT')
        demo.writeOutFrameBuffer()
        print('BUFFER WRITE: COMPLETE')
    # iteration finished. New block
    print('\n')
endTime = time.time()
# final frame buffer flush
demo.writeOutFrameBuffer()
# stitch to gif
# demo.renderGIF()

# Report Results
#os.system('xdg-open out.mp4')
print('time: ',  endTime - startTime)
