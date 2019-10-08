# -*- coding: utf-8 -*-
import pygame
from NeuralNet import network
import sys
import player
import init
import drawing
import connection
import numpy as np
from copy import deepcopy

clock = pygame.time.Clock()

game_size = np.array([100,100])
startpos = np.array([[game_size[0]//4,game_size[1]//4],[3*game_size[0]//4,3*game_size[1]//4]
    ,[game_size[0]//4,3*game_size[1]//4],[3*game_size[0]//4,game_size[0]//4]])
startvel = np.array([[0,1],[0,-1],[1,0],[-1,0]])
startdir = np.array(['down','up','right','left'])

nbots = 2
n_winner = 1
nhumans = 0
N_dir = 8 #number of rays
ninputs = 6*(nbots+nhumans) + N_dir
noutputs = 4
startnet = None
full_nets = True

init.init_game()

def reset_level(game_state,player_list,give_up,ndead):
    #TODO: move chunk from game_loop
    #TODO: set arguments in level class?
    #TODO: needs access to player_list, ndead, game_state
    return
    
def read_inputs(human_list,quitting,ndead,draw_game,inputs,decision,give_up):
    #TODO: same as reset_level, move code, mind variables
    #level and parameter class?
    return

def game_loop():
    #game layer
    best_bot = None
    quitting = False
    draw_game = True
    speed = 1

    #generate first gen of players
    player_list,bot_list,human_list = connection.first_generation(nbots,nhumans
        ,ninputs,noutputs,startnet,full_network=full_nets)
    while not quitting:
        ###GENERATION/GAME LAYER###
        #reset level
        ndead = 0
        game_state = np.zeros(game_size,dtype=int)
        give_up = False
        for ii, plyr in enumerate(player_list):
            plyr.pos = startpos[ii,:]
            plyr.vel = startvel[ii,:]
            plyr.dir = startdir[ii]
            plyr.dead = False
            plyr.score = 10 #start with high score, changed when crashed
            game_state[plyr.pos[0],plyr.pos[1]] = plyr.index
            
        init.gameDisplay.fill((255,255,255))
        activerects = []
        while ndead < len(player_list) - 1:
            ###TICK LAYER###

            inputs = connection.look(game_state,player_list,N_dir)
            decision = connection.think(inputs,player_list)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quitting = True
                    #make loop condition false
                    ndead = len(player_list) + 5
                    
                if event.type == pygame.KEYDOWN:
                    if nhumans > 0:
                        if event.key == pygame.K_UP:
                            human_list[0].turn('up')
                        if event.key == pygame.K_DOWN:
                            human_list[0].turn('down')
                        if event.key == pygame.K_LEFT:
                            human_list[0].turn('left')
                        if event.key == pygame.K_RIGHT:
                            human_list[0].turn('right')
                    if nhumans > 1:
                        if event.key == pygame.K_w:
                            human_list[1].turn('up')
                        if event.key == pygame.K_s:
                            human_list[1].turn('down')
                        if event.key == pygame.K_a:
                            human_list[1].turn('left')
                        if event.key == pygame.K_d:
                            human_list[1].turn('right')

                    if event.key == pygame.K_PERIOD:
                        if speed == 128:
                            speed = 1
                        else:
                            speed = 128
                            
                    if event.key == pygame.K_COMMA:
                        if speed == 1/128.:
                            speed = 1
                        else:
                            speed = 1/128.

                    if event.key == pygame.K_g:
                        draw_game = not draw_game
                    if event.key == pygame.K_r:
                        give_up = True
                    if event.key == pygame.K_p:
                        print(inputs[0])
                        print(decision)

            for plyr in player_list:
                if plyr.dead:
                    continue
                plyr.pos = plyr.pos + plyr.vel
                plyr.pos = plyr.pos % game_size
                if game_state[plyr.pos[0],plyr.pos[1]] != 0:
                    plyr.dead = True
                    #plyr.pos = np.array([-1,-1])
                    plyr.vel = np.array([0,0])
                    plyr.dir = 'dead'
                    ndead += 1
                    #giving scores as current dead players ranks them correctly
                    plyr.score = ndead
                else:
                    game_state[plyr.pos[0],plyr.pos[1]] = plyr.index

	        #END OF TICK LAYER
            if give_up:
                #reset level but don't quit
                break
            if draw_game:
                #init.gameDisplay.fill((255,255,255))
                #prevrects = deepcopy(activerects)
                drawing.draw_game_fast(game_state,player_list) #level & players
                #if nbots > 0:
                    #drawing.draw_net(bot_list[0].net)
                pygame.display.update()

            clock.tick(30*speed)

        #END OF GEN LAYER, dont mutate if quitting
        #Mutation stuff goes here
        if nbots > 0:
            bot_list = connection.next_generation(bot_list,n_winner)
            #TODO: link player_list & bot list
            player_list = bot_list + human_list
                
    '''
    #END OF GAME LAYER, save best bot
    svweights = []
    svfnodes = []
    svtnodes = []
    svnoden = []
    svnodel = []
    for l in range(best_bot.net.layers):
        for conn in best_bot.net.connectionList[l]:
            svweights.append(conn.weight)
            svfnodes.append(conn.fromNode.nodeNo)
            svtnodes.append(conn.toNode.nodeNo)
    
        for node in best_bot.net.nodeList[l]:
            svnodel.append(node.layer)
            svnoden.append(node.nodeNo)
    
    svconn = np.array([svweights,svfnodes,svtnodes])
    svconn = svconn.T
    svnodes = np.array([svnoden,svnodel])
    svnodes = svnodes.T
    if startnet is not None and False:
        np.savetxt(startnet+'_c.txt',svconn,delimiter='\t')
        np.savetxt(startnet+'_n.txt',svnodes,delimiter='\t')
    else:
        np.savetxt('./saves/best_bot_new_c.txt',svconn,delimiter='\t')
        np.savetxt('./saves/best_bot_new_n.txt',svnodes,delimiter='\t')
    '''
game_loop()    
pygame.quit()
sys.exit()
