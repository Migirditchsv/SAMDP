#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""


class testClass(object):
    def __init__(self,_whole, _double, _fxn):
        
        self.whole = _whole
        self.double = _double
        self.fxn = classmethod(_fxn)
        print("made")
        
    def fxn(self, num):
        pass
        
def testFxn(self, num):
    print('testfxn.num= ' % num)
    print('testfxn self.double= ' % self.double)
    
    
# do it.
    
test = testClass(3, 1.3, testFxn)

test.fxn(666)
