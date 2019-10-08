import numpy as np
import player
from NeuralNet import network
from copy import deepcopy

def first_generation(nbots,nhumans,ninputs,noutputs,startnet,full_network=False):
    #generate first gen of players
    bot_list = []
    human_list = []
    for ii in range(1,nbots+1):
        pbuf = player.Player(index=ii,human=False,ninputs=ninputs,noutputs=noutputs,full_network=full_network)
        bot_list.append(pbuf)

        if startnet is not None:
            bot_list[len(bot_list) - 1].net = network.load_net(startnet)
            bot_list.mutate()

    for jj in range(1,nhumans+1):
        pbuf = player.Player(index=nbots+jj,human=True,ninputs=ninputs,noutputs=noutputs)
        human_list.append(pbuf)

    player_list = bot_list + human_list    

    return player_list,bot_list,human_list

def next_generation(bot_list,n_winner):
    scores = np.array([])
    for plyr in bot_list:
        scores = np.append(scores,plyr.score)

    scoreidx = np.argsort(scores)[::-1]

    c_key = ['red','blue','green','gray']
    print(f'{c_key[np.argmax(scores)]} player wins!')
    new_list = []
    #leave n_winner bots alone
    for ii in range(n_winner):
        winner = bot_list[scoreidx[ii]]
        new_list.append(deepcopy(winner))
        new_list[ii].index = ii + 1
    #replace others with copy of a winner
    for ii in range(n_winner,len(bot_list)):
        winner = new_list[ii % n_winner]
        new_list.append(deepcopy(winner))
        new_list[ii].index = ii + 1
        new_list[ii].net.mutate()

    return new_list

def look(game_state,player_list,N_dir):
    #create inputs from gamestate
    '''
    inputs are:
        player x,y,vx,vy for all players, self marked
        distance to nearest wall in N directions
    '''
    nplayer = len(player_list)
    input_len = player_list[0].net.looknodes
    inputs = np.zeros((nplayer,input_len))
    ppos = np.zeros((nplayer,2))
    pdir = np.zeros((nplayer,4))
    game_size = np.array(game_state.shape)

    #up,down,right,left,ur,dr,ul,dl
    dir_key = np.array([[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]])
    dir_key = dir_key[:N_dir]

    #set up pos & vel arrays
    for ii,plyr in enumerate(player_list):
        ppos[ii,:] = plyr.pos
        #dead will have 0 for all directions
        pdir[ii,0] = int(plyr.dir == 'up')
        pdir[ii,1] = int(plyr.dir == 'down')
        pdir[ii,2] = int(plyr.dir == 'left')
        pdir[ii,3] = int(plyr.dir == 'right')

    #TODO: some serious vectorisation
    for ii,plyr in enumerate(player_list):
        if plyr.human:
            #dont calculate inputs for human
            continue
        #keep current player first
        pbuf = np.roll(ppos,-ii,axis=0)
        dbuf = np.roll(pdir,-ii,axis=0)
        inputs[ii,:2*nplayer] = (pbuf/game_size[None,:]).flatten()
        inputs[ii,2*nplayer:6*nplayer] = dbuf.flatten()

        #for each direction, 
        for d,dir in enumerate(dir_key):
            for step in range(1,game_state.shape[0]):
                pos = (plyr.pos + dir*step) % game_state.shape
                inputs[ii,6*nplayer+d] = step/game_size[0]
                if game_state[pos[0],pos[1]] != 0:
                    break
    return inputs

def think(inputs,player_list):
    noutputs = 4 #up,down,left,right
    nplayer = len(player_list)
    decisions = np.zeros((len(player_list),noutputs))
    for ii,plyr in enumerate(player_list):
        if plyr.human:
            #don't try to propogate human network
            continue

        dbuf = plyr.net.propogate(inputs[ii,:])
        decisions[ii,:] = dbuf

        #don't act below threshold
        if dbuf.max() <= 0:
            continue

        dir = np.argmax(dbuf)

        if dir == 0:
            plyr.turn('up')
            #print(f'bot {plyr.index} turned up')
        if dir == 1:
            plyr.turn('down')
            #print(f'bot {plyr.index} turned down')
        if dir == 2:
            plyr.turn('left')
            #print(f'bot {plyr.index} turned left')
        if dir == 3:
            plyr.turn('right')
            #print(f'bot {plyr.index} turned right')

    return decisions
