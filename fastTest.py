#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""
import samdp
import os  # play gif at end

epsilon = 0.001

stateSpace = samdp.stateSpaceGenerator(20, 0.5)

actionPrimatives = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                    (-1, -1), (1, 1), (-1, 1), (1, -1)]

transitionModel = samdp.transitionModelGenerator(
    stateSpace, actionPrimatives, 0.2)

demo = samdp.SAMDP(stateSpace, actionPrimatives, transitionModel)
demo.renderFrame()

while demo.maxDifference > epsilon:

    demo.hybridIterationStep()
    stepNum = demo.solverIterations
    hold = demo.valueFunction

demo.renderFrame()

# final frame buffer flush
demo.writeOutFrameBuffer()
# stitch to gif
demo.renderGIF()

# play the resulting gif
os.system('xdg-open out.mp4')
