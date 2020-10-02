#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import autoTest as at
print('autoTest Active')
size = 25
at.autoTest(environmentSize=size, avgCostTrials=1,
            mfptRank=1, dijkstraSeed=1)
at.autoTest(environmentSize=size, avgCostTrials=1,
            mfptRank=1, dijkstraSeed=0)
at.autoTest(environmentSize=size, avgCostTrials=1,
            mfptRank=0, dijkstraSeed=1)
at.autoTest(environmentSize=size, avgCostTrials=1,
            mfptRank=0, dijkstraSeed=0)
