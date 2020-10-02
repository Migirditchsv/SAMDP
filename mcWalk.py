import autoTest as at
print('autoTest Active')
size = 25
at.autoTest(environmentSize=size, mfptRank=1, dijkstraSeed=1)
at.autoTest(environmentSize=size, mfptRank=1, dijkstraSeed=0)
at.autoTest(environmentSize=size, mfptRank=0, dijkstraSeed=1)
at.autoTest(environmentSize=size, mfptRank=0, dijkstraSeed=0)
