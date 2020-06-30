#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""
import time # testing and comparison
import samdp
import matplotlib.pyplot as plt # convergence rate tests
#import os  # play gif at end

#### Controls ####
# side length of square state space
environmentSize = 20

## Dijkstra Controls
# pre-seed value function based on dijkstra distance?
dijkstraSeed =  False

## mfpt Controls
# seed values by mfpt?
mfptSeed = False
# Update states by mfpt ranking every N steps
mfptUpdatePeriod = 9999
# refresh mfpt ranking every N steps. It is expensive.
mfptRefreshPeriod = 6
# top X percent of mfpt scores to put in update list.
mfptUpdateRatio = 0.3
# Number of random starting states to compute each mfpt score from.
mfptRolloutNumber = 30

## Gradient Smoothing Controls
# seed starting values with a gradient minimizaiton?
gradientSeed = False
# Update states by gradient rank every N steps
gradientUpdatePeriod = 999
# refresh value gradient rank every N steps
gradientRefreshPeriod = 1
# update top X percent of states under gradient ranking
gradientUpdateRatio = 0.2

## Global updates
# run a global sweep every X steps.
globalUpdatePeriod = 999999

#### Initializeation ####
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


# #### Compute optimal value / policy
# optimal = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
# # setup
# optimalTimeStamps = []
# optimalAverageSteps = []
# optimalAverageCost = []
# optimal.updateList = optimal.normalSet
# optimalComputeTime = 0.0
# while optimal.maxDifference > optimal.convergenceThresholdEstimate:
    
#     print('test.py: optimal precompute step num:', optimal.solverIterations)
#     optimalStartTime = time.time()
#     optimal.hybridIterationStep()
#     optimalEndTime = time.time()
#     optimalComputeTime += optimalEndTime - optimalStartTime
#     score = optimal.averageCost()
#     # log scores
#     optimalTimeStamps.append( optimalComputeTime )
#     optimalAverageSteps.append( score[0] )
#     optimalAverageCost.append( score[1] )
    
    
# print("Optimal exhaustive benchmark complete \n")
# lines = plt.plot(optimalTimeStamps, optimalAverageSteps,
#                  optimalTimeStamps, optimalAverageCost)
# plt.setp(lines[0])
# plt.setp(lines[1])
# plt.title("Traditional Hybrid Policy Value Iteration Convergence")
# plt.legend(('Average Steps To Goal', 'Average Cost to  Goal'),loc='upper right')
# plt.show()
# plt.draw()

    
# optimal.policy/valueFunction can now be refferenced.
    



#### Test algorithm performance
demo = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
demo.renderFrame()

# Data holders
demoTimeStamps = []
demoAverageSteps = []
demoAverageCost = []

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
    demo.gradientRank(gradientUpdateRatio, displayGradient=True)
    gradientStopTime = time.time()
    print('Gradient run time: ', gradientStopTime - gradientStartTime)
    demo.policyIterationStep()
gradientUpdateList = []

if mfptSeed == True:
    demo.updateList = demo.normalSet
    demo.policyIterationStep()
    demo.renderFrame()
    mfptStart = time.time()
    mfptUpdateList = demo.mfptRank(mfptUpdateRatio, mfptRolloutNumber)
    mfptStop = time.time()
    print('MFPT Run Time: ', mfptStop - mfptStart)
mfptUpdateList = []

# ???read shoubik first!!! demo.policyUpdate() instead of hybrid step

print('pre-processing complete\n')

# Solve
totalTime = 0.0
unconverged = 1
while unconverged:

    # Clock in
    print('test.py: step num:', demo.solverIterations)
    print('test.py: delta value: ', demo.maxDifference, '/',demo.convergenceThresholdEstimate)
    solverIterations = demo.solverIterations
    startTime = time.time()
    
    # partial update usage
    if solverIterations % gradientUpdatePeriod == 0:
        condition = len(gradientUpdateList) == 0
        condition += solverIterations%gradientRefreshPeriod == 0
        if condition:
            gradientUpdateList = demo.gradientRank(gradientUpdateRatio,
                                            displayGradient=True)
        demo.updateList = gradientUpdateList
        print('Gradient Update Queued')
    elif solverIterations % mfptUpdatePeriod == 0:
        condition = len(mfptUpdateList)==0
        condition += solverIterations%mfptRefreshPeriod == 0
        if condition:
            mfptUpdateList = demo.mfptRank(mfptUpdateRatio, mfptRolloutNumber)
        demo.updateList = mfptUpdateList
        print('MFPT Update Queued')
    else:
        demo.updateList = demo.problemSet
        print('Global Update Queued')

    # Give our ranked problem set, update
    demo.hybridIterationStep()
    unconverged = demo.maxDifference > demo.convergenceThresholdEstimate

    
    # Clock out and store results
    endTime = time.time()
    deltaTime = endTime - startTime
    print('test.py: deltaTime: ',deltaTime)
    totalTime += deltaTime
    results = demo.averageCost()
    steps = results[0]
    cost = results[1]
    demoTimeStamps.append(totalTime)
    demoAverageSteps.append(steps)
    demoAverageCost.append(cost)

    # Save conditions
    stepNum = solverIterations
    print('RENDER FRAME CHECK: INIT')
    if stepNum < 20:
        demo.renderFrame(renderValueGradient=True)
    elif stepNum > 20 and stepNum % 3 == 0:
        demo.renderFrame(renderValueGradient=True)
    elif stepNum < 30 and stepNum % 1 == 0:
        demo.renderFrame(renderValueGradient=True)
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

#### Report Results ####
#os.system('xdg-open out.mp4')
# repost as plots
lines = plt.plot(demoTimeStamps, demoAverageSteps,
                 demoTimeStamps, demoAverageCost)
plt.setp(lines[0])
plt.setp(lines[1])
plt.title("Traditional Hybrid Policy Value Iteration Convergence")
plt.legend(('Average Steps To Goal', 'Average Cost to  Goal'),loc='upper right')
plt.show()
plt.draw()

# final frame buffer flush
demo.writeOutFrameBuffer()
# stitch to gif
# demo.renderGIF()


