# -*- coding: utf-8 -*-
'''
Defines global variables, contains initalisation functions
'''
import pygame

class _imlist():
    def __init__(self):
        self.images = []
        self.names = []
    def add_image(self,fname):
        self.names.append(fname.split('.')[0])
        self.images.append(pygame.image.load(fname))
   
display_width = 600
display_height = 600
gameDisplay = None
Imlist = _imlist()
    
def load_ims(iml):
    #add ims here
    iml.add_image('./example.png')
    
def init_game():
    pygame.init()

    global Imlist, gameDisplay

    gameDisplay = pygame.display.set_mode((display_width,display_height))

    pygame.display.set_caption('The Second Test')
    #load_ims(Imlist)
