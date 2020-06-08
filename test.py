#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""
import samdp
import os  # play gif at end

# Controls
# run mfpt analysis every X steps. It is expensive.
mfptRefreshPeriod = 9999999
# top X percent of mfpt scores to put in update list.
mfptUpdateRatio = 0.3
# Number of random starting states to compute each mfpt score from.
mfptRolloutNumber = 10
# run a global sweep every X steps.
globalSweepPeriod = 1

# Initialize Environment
stateSpace = samdp.stateSpaceGenerator(10, 0.5)

actionPrimatives = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                    (-1, -1), (1, 1), (-1, 1), (1, -1)]

transitionModel = samdp.transitionModelGenerator(
    stateSpace, actionPrimatives, 0.2)

demo = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
demo.renderFrame()

# pre-compute
demo.DijkstraValueSeed()
demo.updateList = demo.normalSet
demo.policyIterationStep()
demo.mfpt(mfptUpdateRatio, mfptRolloutNumber)
# ???read shoubik first!!! demo.policyUpdate() instead of hybrid step

# Solve
while demo.maxDifference > demo.convergenceThresholdEstimate:

    # update
    print('test.py: step num:', demo.solverIterations, '\n')
    print('test.py: delta value: ', demo.maxDifference, '\n')

    # mfpt usage
    if demo.solverIterations % mfptRefreshPeriod == 0:
        mfptUpdateList = demo.mfpt(mfptUpdateRatio, mfptRolloutNumber)

    if demo.solverIterations % globalSweepPeriod == 0:
        demo.updateList = demo.normalSet
    else:
        demo.updateList = mfptUpdateList

    # update
    demo.hybridIterationStep()
    stepNum = demo.solverIterations

    # Save conditions
    if stepNum < 20:
        demo.renderFrame()
    elif stepNum > 20 and stepNum % 3 == 0:
        demo.renderFrame()
    elif stepNum < 30 and stepNum % 4 == 0:
        demo.renderFrame()

    if demo.frameBuffer % 30 == 0:
        # One unwritten frame wastes ~10mb of ram, but writing is slow
        demo.writeOutFrameBuffer()

# final frame buffer flush
demo.writeOutFrameBuffer()
# stitch to gif
demo.renderGIF()

# Report Results
os.system('xdg-open out.mp4')
