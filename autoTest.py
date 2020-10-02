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
plottingOn = True  # Skip making animations for faster results
computeAverageCost = True  # Make cost and convergence plots. Expensive


def autoTest(environmentSize=12, avgCostTrials=3, avgCostSamplePeriod=1, dijkstraSeed=0, mfptRank=0):

    # Labels
    # prefix
    if mfptRank:
        prefix = 'MFPT Ranked Update '
    else:
        prefix = 'Raster Ordered Update '
    # suffix
    if dijkstraSeed:
        suffix = 'With Seeding'
    else:
        suffix = 'Without Seeding'

    runTitle = prefix+suffix
    runPath = (prefix+suffix).replace(' ', '')
    print("Beginning Run: ", runTitle, '\nOn filepath: ', runPath)
    convergencePlotTitle = runTitle+'\n' + \
        'Policy Convergence vs. Time'
    costPlotTitle = runTitle+'\n'+'Average Utility of Markov Chain Versus Time'

    # Initialize Environment
    stateSpace = samdp.stateSpaceGenerator(environmentSize)

    # When defining action primatives, the iteration schemes break ties between
    # expected utility of actions by the order they appear in the primatives list.
    # null actions (0,0), should therefore always be listed LAST
    directional8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
                    (-1, -1), (1, 1), (-1, 1), (1, -1), (0, 0)]

    directional4 = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

    actionPrimatives = directional8

    # Create key data objects
    demo = samdp.SAMDP(stateSpace, actionPrimatives)
    # Create dir
    demo.createDataDir(runPath)
    demo.renderFrame(title=runTitle)
    timeFrames = []
    deltaPolicyFrames = []
    avgCostFrames = []
    previousPolicy = demo.policy.copy()

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
        if not mfptRank:
            demo.policyIterationStep()
        else:  # Do mfpt ranked update
            demo.mfptPolicyIteration()

        # Check For Convergence
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
        if computeAverageCost:
            (deltaPolicy, cost) = demo.averageCost(
                previousPolicy, avgCostTrials, avgCostSamplePeriod)
            timeFrames.append(totalTime)
            deltaPolicyFrames.append(deltaPolicy)
            avgCostFrames.append(cost)

        # Save conditions
        if plottingOn:
            print('RENDER FRAME CHECK: INIT')
            demo.renderFrame(title=runTitle)
            print('RENDER FRAME CHECK: COMPLETE')

        if demo.frameBuffer % 60 == 0:
            # One unwritten frame wastes ~10mb of ram, but writing is slow
            print('BUFFER WRITE: INIT')
            demo.writeOutFrameBuffer(runPath)
            print('BUFFER WRITE: COMPLETE')
        # iteration finished. New block
        print('\n')

    # Complete. Report
    print('COMPLETE')
    print('run time: ',  totalTime, '\n')

    # final frame buffer flush
    if plottingOn:
        demo.writeOutFrameBuffer(runPath)

    # Generate convergence plot
    if computeAverageCost:
        plt.plot(timeFrames, deltaPolicyFrames,
                 marker="x")
        plt.title(convergencePlotTitle)
        plt.xlabel('Elapsed Time [seconds]')
        plt.ylabel('Ratio of states with altered actions')
        # plt.legend(('Change in policy', 'Average Cost to  Goal'), loc='upper right')
        plt.savefig('RenderingFrames/'+runPath +
                    '/'+runPath+'convergence.png')
        plt.close()

        # Genreate cost plot
        plt.plot(timeFrames, avgCostFrames, marker='x')
        plt.title(costPlotTitle)
        plt.xlabel('Elapsed Time [seconds]')
        plt.ylabel('Average utility over all states')
        plt.savefig('RenderingFrames/'+runPath+'/'+runPath+'cost.png')
        plt.close()

    # stitch to gif
    if plottingOn:
        demo.renderGIF(runPath)
