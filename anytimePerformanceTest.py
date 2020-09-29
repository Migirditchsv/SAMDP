#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""
import time  # testing and comparison
import samdp
import math as m  # floor
import matplotlib.pyplot as plt  # convergence rate tests
# import os  # play gif at end


print("AnytimePerformaceTest Begin")
#### Controls ####
# side length of square state space
environmentSize = 25
# title of data for this run
convergencePlotTitle = 'Convergence With Exclusive Full MFPT Ranked Updates'


# Dijkstra Controls
# pre-seed value function based on dijkstra distance?
dijkstraSeed = False

# mfpt Controls
# seed values by mfpt?
mfptSeed = False
# run an mfpt ranked update every N iterations
mfptUpdatePeriod = 1
# run mfpt analysis every X mpft updates. refresh every 3-5 iterations
mfptRefreshPeriod = 4
# top X percent of mfpt scores to put in update list.
mfptUpdateRatio = 1.0

# Gradient Smoothing Controls
# seed starting values with a gradient minimizaiton?
gradientSeed = False
# Run a gradient ranked update every N steps
gradientUpdatePeriod = 9999
# Run value gradient ranked refresh every N steps
gradientRefreshPeriod = 1
# update top X percent of states under gradient ranking
gradientUpdateRatio = 0.7

# Global updates
# run a global sweep every X steps. Overrides other concerent update types
globalUpdatePeriod = 999999

#### Initializeation ####
stateSpace = samdp.stateSpaceGenerator(environmentSize)

# When defining action primatives, the iteration schemes break ties between
# expected utility of actions by the order they appear in the primatives list.
# null actions (0,0), should therefore always be listed LAST
directional8 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (-1, -1), (1, 1), (-1, 1), (1, -1)]

directional4 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

actionPrimatives = directional8

transitionModel = samdp.transitionModelGenerator(
    stateSpace, actionPrimatives, 0.2)


# Data holders
demoTimeStamps = []
demoAverageSteps = []
demoAverageCost = []

# Test algorithm performance
demo = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
demo.renderFrame()
# pre-performance eval
results = demo.averageCost(3)
steps = results[0]
cost = results[1]
demoTimeStamps.append(0)
demoAverageSteps.append(steps)
demoAverageCost.append(cost)

print('Test initalized\n')
# pre-compute
if dijkstraSeed == True:
    # run dijkstra
    dijkstraStart = time.time()
    demo.DijkstraValueSeed()
    demo.policyIterationStep()
    dijkstraStop = time.time()
    print('Dijkstra Run Time: ', dijkstraStop - dijkstraStart)

if gradientSeed == True:
    gradientStartTime = time.time()
    demo.gradientRank(gradientUpdateRatio)
    gradientStopTime = time.time()
    print('Gradient run time: ', gradientStopTime - gradientStartTime)
    demo.policyIterationStep()

if mfptSeed == True:
    demo.updateList = demo.normalSet
    demo.policyIterationStep()
    demo.renderFrame()
    mfptStart = time.time()
    mfptUpdateList = demo.mfptRank(mfptUpdateRatio, mfptRolloutNumber)
    mfptStop = time.time()
    demp.renderMFPT()
    print('MFPT Run Time: ', mfptStop - mfptStart)


print('pre-processing complete\n')

# Solve
totalTime = 0.0
unconverged = 1
while unconverged:

    # update
    print('test.py: step num:', demo.solverIterations)
    print('test.py: delta value: ', demo.maxDifference,
          '/', demo.convergenceThresholdEstimate)

    # Clock in
    startTime = time.time()
    iteration = demo.solverIterations

    # partial update usage
    if (iteration % globalUpdatePeriod == 0):
        demo.updateList = demo.problemSet
        print('Global Update Queued')
    elif iteration % gradientUpdatePeriod == 0:
        if (iteration // gradientUpdatePeriod) % gradientRefreshPeriod == 0:
            gradientUpdateList = demo.gradientRank(gradientUpdateRatio)
        demo.updateList = gradientUpdateList
        print('Gradient Update Queued')
    elif iteration % mfptUpdatePeriod == 0:
        if (iteration // mfptUpdatePeriod) % mfptRefreshPeriod == 0:
            mfptUpdateList = demo.mfptRank(mfptUpdateRatio)
        demo.updateList = mfptUpdateList
        demo.renderMFPT()
        print('MFPT Update Queued')
    else:  # default to global update
        demo.updateList = demo.problemSet
        print('Global Update Queued')

    # Give our ranked problem set, update
    demo.hybridIterationStep()
    unconverged = demo.maxDifference > demo.convergenceThresholdEstimate

    # Clock out and store results
    # time out
    endTime = time.time()
    deltaTime = endTime - startTime
    print('test.py: deltaTime: ', deltaTime)
    totalTime += deltaTime
    # performance eval
    results = demo.averageCost()
    steps = results[0]
    cost = results[1]
    demoTimeStamps.append(totalTime)
    demoAverageSteps.append(steps)
    demoAverageCost.append(cost)
    # render gradient
    #demo.renderValueGradient()

    # Save conditions
    stepNum = iteration
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
print('run time: ',  totalTime, '\n')

#### Report Results ####
#os.system('xdg-open out.mp4')
# clean data
demoTimeStamps.pop(1)
demoAverageCost.pop(1)
demoAverageSteps.pop(1)
lines = plt.plot(demoTimeStamps, demoAverageSteps,
                 demoTimeStamps, demoAverageCost)
plt.setp(lines[0])
plt.setp(lines[1])
plt.title(convergencePlotTitle)
plt.legend(('Average Steps To Goal', 'Average Cost to  Goal'), loc='upper right')
plt.savefig('RenderingFrames/convergence.png')

# final frame buffer flush
demo.writeOutFrameBuffer()
# stitch to gif
demo.renderGIF()
