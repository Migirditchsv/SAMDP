#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:16:40 2020

@author: sam
"""
import samdp

    # Controls
    environmentSize = 20
    
    # Initialize Environment
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
    
    testo = samdp.SAMDP(stateSpace,actionPrimatives,transitionModel)
