#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:37:48 2023

@author: tristan
"""


import numpy as np
import warnings


#code of each particles in the swarm
class Particle:
 
    def __init__(self,dimension):
        self.pos=np.random.random(dimension)
        self.velocity=0.1*np.random.random(dimension)
        self.constraints(dimension)
        self.pBest=self.pos.copy()
        return
 
    def updatePositions(self):
        self.pos += self.velocity
        return
 
    def updateVelocities(self,dimension, gBest,iter_current=0,iter_max=100,c1=1,c2=1,w_max=1,w_min=0.1):
        r1=np.random.random(dimension)
        r2=np.random.random(dimension)
        social = c1*r1*(gBest - self.pos)
        cognitive = c2*r2*(self.pBest - self.pos)
        
        #The inertia depends on how far we are in the exploration phase.
        #At the beginning, we favorise large jumps to see a broader region of the configuration space
        #At the end, we restrict the size of jumps to have a finer exploration near minimas of the optimisation function
        w=(w_max - w_min)*(iter_current)/iter_max + w_min
        self.velocity=w*self.velocity + social + cognitive
        return
 
    def constraints(self,dimension):
        #To ensure that the constraints are satisfied
        
        satisfied=False
        k=0
        
        while (satisfied==False or abs(self.pos.sum()-1)>0.005) and k<50:
            
            self.pos=(self.pos-(np.dot(self.pos,np.ones(dimension))-1)*np.ones(dimension)/dimension).copy()
            
            satisfied=True
            for i in range(len(self.pos)):
                if self.pos[i]<0:
                    satisfied=False
                    self.pos[i]=0.1*np.random.random(1)
                elif self.pos[i]>1:
                    satisfied=False
                    self.pos[i]=1-0.1*np.random.random(1)
                    
            #If after 30 iterations, we still didn't reach a feasible configuration, consider a configuration well within the configuration space
            #Here the barycentre of the space plus some small perturbation
            if k>40: 
                # print("big k")
                self.pos=np.ones(dimension)/dimension + (np.random.random(dimension)-0.5)/(20*dimension)
            
            # print(k,satisfied,self.pos,self.pos.sum())
            
            if k>=48:
                self.pos=np.ones(dimension)/dimension
            k+=1
            
        
        if k==50: 
            warnings.warn(f"An unfeasible solution that can't be projected back to a feasible solution has been found \n {self.pos} \n {self.pos.sum()} {self.pos.min()} {self.pos.max()}, {k}")
            self.pos=np.ones(dimension)/dimension
            
        return
    
    def displayPosition(self):
        print(f"Particle position: {self.pos}, pBest: {self.pBest}")
    
    
#particle swarm optimization algorithm
class ParticleSwarmOptimizer:
 
    def __init__(self,dimension=2,swarm_size=20,iterations=10,c1=1.,c2=1.,w_min=0.5,w_max=0.9):
        """
        Creation of the particle swarm optimiser

        Parameters
        ----------
        dimension : int, optional
            Number of variables forming the problem. The default is 2.
        swarm_size : int, optional
            Number of particles consisting the swarm. The default is 20.
        iterations : int, optional
            Iteration number. The default is 10.
        c1 : float, optional
            social coefficient. The default is 1..
        c2 : float, optional
            cognitive coefficient. The default is 1..
        w_min : float, optional
            inertia of the particles at the end of the iterations. The default is 0.5.
        w_max : float, optional
            inertia of the particles at the beginning of the iterations. The default is 0.9.

        Returns
        -------
        None.

        """
        
        self.swarm_size=swarm_size
        self.dimension=dimension
        self.iterations=iterations
        self.c1=c1
        self.c2=c2
        self.w_min=w_min
        self.w_max=w_max
        self.swarm=[]
        
        
        for h in range(self.swarm_size):
            self.swarm.append(Particle(self.dimension))
            
        # self.displayPositions()
        return
    

 
    def optimize(self,func,verbose=False,args=dict()):
        """
        Optimisation of the function func under the constraint that the sum of all the variables is equal to one

        Parameters
        ----------
        func : function
            Function to maximise.
        verbose : boolean, optional
            gives informations during the execution of the optimisation. The default is False.
        args : dict, optional
            Arguments needed for the computation of the function to maximise. The default is dict().

        Returns
        -------
        solution : numpy array
            Optimal solution found.

        """
        
        
        # Initialisation
        gBest = self.swarm[0].pos
        for i in range(self.swarm_size):
            if func(self.swarm[i].pBest,args) > func(gBest,args):
                gBest = self.swarm[i].pBest.copy()  
        # print(f"Initialisation: {gBest}")
        for i in range(self.iterations):
           

            #Update the position of each paricle
            for j in range(self.swarm_size):
                self.swarm[j].updateVelocities(self.dimension,gBest,iter_current=i,iter_max=self.iterations,c1=self.c1,c2=self.c2,w_max=self.w_max,w_min=self.w_min)
                self.swarm[j].updatePositions()
                self.swarm[j].constraints(self.dimension)
                
                
            # self.displayPositions()
                
            #Update the personal best positions of each particle
            for k in range(self.swarm_size):
                if func(self.swarm[k].pos,args) > func(self.swarm[k].pBest,args):
                    self.swarm[k].pBest = self.swarm[k].pos.copy()
                    
                #Find if one of the best position of each particle of the swarm outperforms the current global best position
                #print(func(self.swarm[k].pBest,args) , func(gBest,args))
                if func(self.swarm[k].pBest,args) > func(gBest,args):
                    gBest = self.swarm[k].pBest.copy()
                    
            solution = gBest
            
            # self.displayPositions()
            if verbose:
                print(f"iteration: {i+1}/{self.iterations}, best solution:  {func(gBest,args)}")
            
        return solution
    
    def displayPositions(self):
        #displays the position of all the particles of the swarm
        for s in self.swarm:
            s.displayPosition()
 