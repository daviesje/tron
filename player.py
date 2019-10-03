# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:34:32 2019

@author: jed12
"""
from NeuralNet import network
import numpy as np

class Player():
    def __init__(self,index,human,ninputs,noutputs,full_network=False):
        if not human:
            if full_network:
                self.net = network.full_network(ninputs,noutputs)
                self.net.mutate()
            else:
                self.net = network.Network()
                self.net.fresh_start(ninputs,noutputs)
                self.net.mutate()
        self.pos = np.array([0,0])
        self.vel = np.array([0,1])
        self.dead = False
        self.index = index
        self.human = human
        self.score = 0
        self.dir = 'up' #redundant with vel

    def turn(self,direction):
        if self.dead:
            return
        if direction == 'up':
            if self.dir in ['left','right']:
                self.dir = 'up'
                self.vel = [0,-1]
        if direction == 'down':
            if self.dir in ['left','right']:
                self.dir = 'down'
                self.vel = [0,1]
        if direction == 'left':
            if self.dir in ['up','down']:
                self.dir = 'left'
                self.vel = [-1,0]
        if direction == 'right':
            if self.dir in ['up','down']:
                self.dir = 'right'
                self.vel = [1,0]
