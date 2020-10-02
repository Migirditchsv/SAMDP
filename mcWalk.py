#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import autoTest as at
print('autoTest Active')
size = 40
at.autoTest(environmentSize=size, avgCostTrials=1, avgCostSamplePeriod=13,
            mfptRank=1, dijkstraSeed=1)
at.autoTest(environmentSize=size, avgCostTrials=1, avgCostSamplePeriod=13,
            mfptRank=1, dijkstraSeed=0)
at.autoTest(environmentSize=size, avgCostTrials=1, avgCostSamplePeriod=13,
            mfptRank=0, dijkstraSeed=1)
at.autoTest(environmentSize=size, avgCostTrials=1, avgCostSamplePeriod=13,
            mfptRank=0, dijkstraSeed=0)
