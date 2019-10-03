# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:53:50 2019

@author: jed12
"""
from math import exp
import numpy as np
from numpy.random import rand, randint, randn

def load_net(fname):
    conn_arr = np.loadtxt(fname+'_c.txt',delimiter='\t')
    node_arr = np.loadtxt(fname+'_n.txt',delimiter='\t').astype(int)
    net = Network()
    nodemap = [[],[],[],[],[]]
    for row in node_arr:
        nodemap[row[1]].append(int(row[0]))
        net.add_node(row[1])
    for row in conn_arr:
        for l in range(net.layers):
            for i,n in enumerate(nodemap[l]):
                if n == row[1]:
                    fromnode = net.nodeList[l][i]
                if n == row[2]:
                    tonode = net.nodeList[l][i]
        net.add_connection(fromnode,tonode,row[0])
    return net

class Connection():
    def __init__(self):
        self.connectionno = -1
        self.fromNode = None
        self.toNode = None
        self.weight = 0
        self.layer = 1 #To layer
    def engage(self):
        self.toNode.input += self.fromNode.output*self.weight

class Node():
    def __init__(self,layer=0):
        self.nodeNo = -1
        self.layer = -1
        self.input = 0
        self.output = 0
    #sigmoid input -> output
    def engage(self):
        self.output = 2 / (1 + exp(-self.input)) - 1
        
class Network():
    def __init__(self):
        self.nodeList = [[],[],[],[],[]]
        self.connectionList = [[],[],[],[],[]]
        self.layers = 5
        self.width = 5
        self.noden = 0
        self.looknodes = 0 #normal amount
        self.outnodes = 0 #normal amount
        self.weights_only = False
    def fresh_start(self,ninputs,noutputs):
        self.__init__
        #add inputs + bias
        self.looknodes = ninputs
        self.outnodes = noutputs
        [self.add_node(layer=0) for i in range(ninputs + 1)]
        for i in range(noutputs):
            #add outputs
            node = self.add_node(layer=self.layers-1)
            #add bias - output connections
            self.add_connection(self.nodeList[0][ninputs],node,0)
    def add_node(self,layer):
        newnode = Node()
        newnode.nodeNo = self.noden
        newnode.layer = layer
        self.nodeList[layer].append(newnode)
        self.noden += 1
        return newnode
    def add_connection(self,fromnode,tonode,weight):
        assert fromnode.layer < tonode.layer
        n = len(self.connectionList)
        newcon = Connection()
        newcon.fromNode = fromnode
        newcon.toNode = tonode
        newcon.layer = tonode.layer
        newcon.weight = weight
        newcon.connectionno = n
        self.connectionList[newcon.layer].append(newcon)
        return newcon
    def propogate(self,sights):
        decision = np.zeros(self.outnodes)
        assert(sights.size == self.looknodes)
        self.reset_network()
        for i,lnode in enumerate(self.nodeList[0][:self.looknodes]):
            lnode.output = sights[i]
        #Don't need to engage layer 0 since sights are outputs
        for layer in range(1,self.layers):
            for c in self.connectionList[layer]:
                c.engage()
            for n in self.nodeList[layer]:
                n.engage()
        for i,onode in enumerate(self.nodeList[self.layers-1]):
            decision[i] = onode.output
        return decision
    def reset_network(self):
        for l in range(self.layers):
            for n in self.nodeList[l]:
                n.input = 0
                n.output = 0
        self.nodeList[0][self.looknodes].output = 1
    def mutate(self):
        switch = rand()
        if self.weights_only:
            switch = 0.1
        #90% change weights
        if switch < 0.90:
            change_weights(self)
        #9% add connection between random nodes
        elif switch < 0.99:
            mutate_connection(self)
        #1% add new node by splitting random connection
        else:
            mutate_node(self)

def full_network(ninputs,noutputs):
    net = Network()
    net.fresh_start(ninputs,noutputs)
    net.weights_only = True
    for i in range(1,net.layers-1):
        for w in range(net.width):
            net.add_node(layer=i)
            
    for l in range(net.layers-1):
            for fn in net.nodeList[l]:
                for ft in net.nodeList[l+1]:
                    net.add_connection(fn,ft,0)
                
    return net
            
def get_counts(net):
    nhist = [0]*net.layers
    chist = [0]*net.layers
    nconn = 0
    nnode = 0
    for l in range(net.layers):
        nhist[l] = len(net.nodeList[l])
        chist[l] = len(net.connectionList[l])
            
    nconn = sum(chist)
    nnode = sum(nhist)
        
    return nhist,chist,nnode,nconn

def fully_connected(net):
    maxcon = 0

    nhist,_,_,nconn = get_counts(net)
    for l in range(net.layers-1):
        x = nhist[l]*sum(nhist[l+1:])
        maxcon = maxcon + x
        
    return nconn >= maxcon
    
def change_weights(net):
    for l in range(net.layers):
        for conn in net.connectionList[l]:
            rand2 = rand()
            if rand2 < 0.1:
                conn.weight = 3*(2*rand()-1)
            else:
                conn.weight += (randn()*0.1)
            if conn.weight > 3:
                conn.weight = 3
            elif conn.weight < -3:
                conn.weight = -3

def mutate_connection(net):
    #check if fully connected
    if fully_connected(net):
        change_weights(net)
        return
    nhist,_,nnode,_ = get_counts(net)
    wgt = 2*rand()-1
    sel1 = randint(0,nnode-net.outnodes)
    while sel1 == net.looknodes:
        sel1 = randint(0,nnode-net.outnodes)
    #find fromnode
    layer = 0
    buf = sel1
    while buf >= nhist[layer]:
        buf -= nhist[layer]
        layer += 1
    n1 = net.nodeList[layer][buf]

    #tonode must be in the next layer
    #print(sel1,buf,nhist,layer,nnode)
    sel2 = randint(sel1-buf+nhist[layer],nnode)
    buf = sel2
    layer = 0
    while buf >= nhist[layer]:
        buf -= nhist[layer]
        layer += 1
    n2 = net.nodeList[layer][buf]
        
    net.add_connection(n1,n2,wgt)

def find_splittable(net):
    splittable = [[],[],[],[],[]]
    totals = [0,0,0,0,0]
    for l in range(net.layers):
        for idx,c in enumerate(net.connectionList[l]):
            if c.fromNode.layer < c.layer - 1 and c.fromNode != net.nodeList[0][net.looknodes]:
                splittable[l].append(idx)
        totals[l] = len(splittable[l])
                
    return splittable,totals,sum(totals)
    
def mutate_node(net):  
    #find valid connections to split
    sel,nsplit,ntot = find_splittable(net)
    if ntot == 0:
        mutate_connection(net)
        return
    #select one splittable connection
    buf = randint(0,ntot)
    layer = 0
    while buf >= nsplit[layer]:
        buf -= nsplit[layer]
        layer += 1
    #print(buf,layer,sel,nsplit,ntot)
    #print(net.connectionList)
    conn = net.connectionList[layer][sel[layer][buf]]
           
    fromnode = conn.fromNode
    tonode = conn.toNode
    tolayer = conn.layer
    fromlayer = conn.fromNode.layer

    wgt = conn.weight

    newlayer = randint(fromlayer+1,tolayer)
    #temporary fix, reject new nodes in large layers
    if len(net.connectionList[newlayer]) >= net.width:
        mutate_connection(net)
        return

    #REMOVE CONNECTION        
    net.connectionList[layer].remove(conn)

    #ADD NODE
    #print(f'adding node in connection {sel}, layers {fromlayer,newlayer,tolayer}')
    newnode = net.add_node(newlayer)
    #ADD CONNECTIONS
    net.add_connection(fromnode,newnode,1)
    net.add_connection(newnode,tonode,wgt)
    net.add_connection(net.nodeList[0][net.looknodes],newnode,0)
