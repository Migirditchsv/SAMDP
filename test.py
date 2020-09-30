#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""
import time  # testing and comparison
import matplotlib.pyplot as plt  # convergence rate tests
import samdp
import math as m  # Log scale cost fxn
# import os  # play gif at end

# Controls

# Boundary Conditions
# side length of square state space
environmentSize = 12

# Eval Controls
avgCostTrials = 3  # Number of times to simulate walks from each init state
avgCostSamplePeriod = 1  # Use every Nth state as a starting point
plottingOn = True  # Skip making animations for faster results

# Labels
runTitle = 'mfpt seed False complex 12x12'
convergencePlotTitle = runTitle+'\n' + \
    'Policy Convergence vs. Time'
costPlotTitle = runTitle+'\n'+'Average Utility of Markov Chain Versus Time'

# Seed Settings
# pre-seed value function based on dijkstra distance?
dijkstraSeed = False
mfptSeed = False
gradientSeed = False

# Global Update Settings
# run a global sweep every X steps.
globalUpdatePeriod = 1111

# MFPT Settings
# Run an MFPT ranked update every N steps
mfptUpdatePeriod = 1
# run mfpt analysis every X updates. 3-5 is empriacle optimum
mfptRefreshPeriod = 4 * mfptUpdatePeriod
# top X percent of mfpt scores to put in update list.
mfptUpdateRatio = 1.0

# Gradient Settings
# seed starting values with a gradient minimizaiton?
gradientSeed = False
# Run gradient ranked iteration every N steps
gradientUpdatePeriod = 9999999
# Run value gradient ranked refresh every N updates
gradientRefreshPeriod = 999999 * gradientUpdatePeriod
# update top X percent of states under gradient ranking
gradientUpdateRatio = 0.33


# Initialize Environment
stateSpace = samdp.stateSpaceGenerator(environmentSize)

# When defining action primatives, the iteration schemes break ties between
# expected utility of actions by the order they appear in the primatives list.
# null actions (0,0), should therefore always be listed LAST
directional8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (-1, -1), (1, 1), (-1, 1), (1, -1), (0, 0)]

directional4 = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

actionPrimatives = directional8

transitionModel = samdp.transitionModelGenerator(
    stateSpace, actionPrimatives, 0.2)

# Create key data objects
demo = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
demo.renderFrame()
timeFrames = []
deltaPolicyFrames = []
avgCostFrames = []
previousPolicy = demo.policy.copy()

# pre-performance eval
# (steps,cost) = demo.averageCost(avgCostTrials)
# timeFrames.append(0)
# avgStepFrames.append(steps)
# avgCostFrames.append(cost)

print('Test initalized\n')
# pre-compute
dijkstraTimeCost = 0.0
if dijkstraSeed == True:
    # run dijkstra
    dijkstraStart = time.time()
    demo.DijkstraValueSeed()
    demo.policyIterationStep()
    dijkstraStop = time.time()
    dijkstraTimeCost = dijkstraStop - dijkstraStart
    print('Dijkstra Run Time: ', dijkstraTimeCost)

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
    mfptUpdateList = demo.mfptRank(mfptUpdateRatio)
    mfptStop = time.time()
    print('MFPT Run Time: ', mfptStop - mfptStart)

print('pre-processing complete\n')

# Solve
totalTime = dijkstraTimeCost
mfptUpdateList = []
unconverged = 1
while unconverged:

    iteration = demo.solverIterations
    previousPolicy = demo.policy.copy()

    # update
    print('test.py: step num:', iteration)
    print('test.py: delta value: ', demo.maxDifference,
          ':', demo.convergenceThresholdEstimate)

    # Clock in
    startTime = time.time()

   # Select Update

   # Global update
    if iteration % globalUpdatePeriod == 0:
        demo.updateList = demo.problemSet
        demo.hybridIterationStep()
        # Convergence can only be accurately tested for after a global update.
    # MFPT Update
    elif iteration % mfptUpdatePeriod == 0:
        if (iteration % mfptRefreshPeriod)*len(mfptUpdateList) == 0:
            #print('MFPT RE-RANK: INIT')
            mfptUpdateList = demo.mfptRank(mfptUpdateRatio)
        demo.updateList = mfptUpdateList
        demo.mfptPolicyIteration()
    # Gradient Update
    elif iteration % gradientRefreshPeriod:
        demo.updateList = demo.gradientRank(gradientUpdateRatio)

    # Give our ranked problem set, update
    unconverged = (demo.maxDifference >
                   demo.convergenceThresholdEstimate) and\
        (demo.policy != previousPolicy) and\
        (iteration < 101) or\
        (iteration < 5)

    # Clock out
    endTime = time.time()
    deltaTime = endTime - startTime
    print('test.py: deltaTime: ', deltaTime)
    totalTime += deltaTime

    # Score performance
    (deltaPolicy, cost) = demo.averageCost(
        previousPolicy, avgCostTrials, avgCostSamplePeriod)
    timeFrames.append(totalTime)
    deltaPolicyFrames.append(deltaPolicy)
    avgCostFrames.append(cost)

    # Save conditions
    if plottingOn:
        print('RENDER FRAME CHECK: INIT')
        if iteration < 20:
            demo.renderFrame()
        elif iteration > 20 and iteration % 1 == 0:
            demo.renderFrame()
        elif iteration < 30 and iteration % 1 == 0:
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

# final frame buffer flush
demo.writeOutFrameBuffer()

# Generate convergence plot
plt.plot(timeFrames, deltaPolicyFrames,
         marker="x")
plt.title(convergencePlotTitle)
plt.xlabel('Elapsed Time [seconds]')
plt.ylabel('Ratio of states with altered actions')
#plt.legend(('Change in policy', 'Average Cost to  Goal'), loc='upper right')
plt.savefig('RenderingFrames/convergence.png')
plt.close()

# Genreate cost plot
plt.plot(timeFrames, avgCostFrames, marker='x')
plt.title(costPlotTitle)
plt.xlabel('Elapsed Time [seconds]')
plt.ylabel('Average utility over all states')
plt.savefig('RenderingFrames/cost.png')
plt.close()

# stitch to gif
if plottingOn:
    demo.renderGIF()

# Report Results
#os.system('xdg-open out.mp4')
