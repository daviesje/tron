# -*- coding: utf-8 -*-
'''
Contains classes and functions for drawing and manipulating objects
'''
import pygame
import init
import numpy as np
from NeuralNet import network

#0 for no player, 1 - 4 for player colors
color_key = np.array([[255,255,255],[127,0,0],[0,0,127],[0,127,0],[63,63,63]])
pygame.font.init()
myfont = pygame.font.SysFont('Arial',24)

def draw_scores(ims,scores,lnum):
    surf = myfont.render('level '+str(lnum+1),False,(0,0,0))
    init.gameDisplay.blit(surf,(init.display_width/2,64))
    for ii in range(len(ims)):
        init.gameDisplay.blit(pygame.transform.scale2x(init.Imlist.images[ims[ii]]),(64*(1+ii),32))
        surf = myfont.render(str(scores[ii]),False,(0,0,0))
        init.gameDisplay.blit(surf,(64*(1+ii),96))
    
def draw_game(game_state,player_list):
    color_arr = color_key[game_state,:]
    width = init.display_width//color_arr.shape[0]
    height = init.display_height//color_arr.shape[1]
    #make actual player positions brighter
    for plyr in player_list:
        if plyr.dead:
            color_arr[plyr.pos[0],plyr.pos[1]] = np.array([0,0,0])
        else:
            color_arr[plyr.pos[0],plyr.pos[1]] = color_key[plyr.index,:] + np.array([50,50,50])
    #pygame draw colors
    for ii in range(color_arr.shape[0]):
        for jj in range(color_arr.shape[1]):
            left = ii*width
            top = jj*height
            rect = pygame.Rect(left,top,width,height)
            color = color_arr[ii,jj]
            init.gameDisplay.fill(color,rect)
    return

def draw_game_fast(game_state,player_list):
    width = init.display_width//game_state.shape[0]
    height = init.display_height//game_state.shape[1]

    for plyr in player_list:
        left = plyr.pos[0]*width
        top = plyr.pos[1]*height
        rect = pygame.Rect(left,top,width,height)
        left_prev = left - plyr.vel[0]*width
        top_prev = top - plyr.vel[1]*height
        rect_prev = pygame.Rect(left_prev,top_prev,width,height)
        color_prev = color_key[plyr.index]
        if plyr.dead:
            color = np.array([0,0,0])
        else:
            #make actual player positions brighter
            color = color_key[plyr.index,:] + np.array([50,50,50])

        init.gameDisplay.fill(color_prev,rect_prev)
        init.gameDisplay.fill(color,rect)
    return

def draw_net(net):
    screencen = init.display_width/2.
    netcen = int(net.layers/2)
    nhist,chist,nnode,nconn = network.get_counts(net)
    max_nodes = max(nhist)

    nodex = np.zeros(nnode,dtype=int)
    nodey = np.zeros(nnode,dtype=int)
    offset = [0,-8,8,-8,0]
    for l in range(net.layers):
        for ii,node in enumerate(net.nodeList[l]):
            nodex[node.nodeNo] = int(screencen + (node.layer - netcen)*64)
            nodey[node.nodeNo] = int(16*(ii+1)*max_nodes/(nhist[l]+1)
            + offset[node.layer])
            
    for l in range(net.layers):
        for ii,conn in enumerate(net.connectionList[l]):
            red = 223 - min(max([0,conn.weight])*96,223)
            blue = 223 - min(abs(conn.weight)*96,223)
            green = 223 - min(-min([0,conn.weight])*96,223)
            pygame.draw.line(init.gameDisplay,(red,green,blue)
                            ,(nodex[conn.fromNode.nodeNo]
                            ,nodey[conn.fromNode.nodeNo])
                            ,(nodex[conn.toNode.nodeNo]
                            ,nodey[conn.toNode.nodeNo])
                            ,2)
                            
    for ii in range(len(nodex)):
        pygame.draw.circle(init.gameDisplay,(0,0,0),(nodex[ii],nodey[ii]),5)
