#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:00:11 2023

@author: tristan
"""

import PSO

    
"""
Simple function to optimise in order to test the particle swarm otpimiser
"""
    
def function_to_minimise(r,*args):
    return 1-(r[0]**2+r[1]**2)
 

pso = PSO.ParticleSwarmOptimizer(swarm_size=200,dimension=2,iterations=10)
print(pso.optimize(function_to_minimise,verbose=True))



