#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:27:14 2020

@author: sam
"""

a = 0
b = 20
while(b>=0):
    if(b%6==0):
        a = a - b
    else:
        a = a + 5
    b= b - 2
print(a)
