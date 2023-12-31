import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
pygame.init()
font = pygame.font.Font(None,25)

# Reset 
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Point = namedtuple('Point','x , y')

BLOCK_SIZE=20
SPEED = 40
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

class SnakeGameAI:
    def __init__(self,w=BLOCK_SIZE*30,h=BLOCK_SIZE*20):
        self.w=w
        self.h=h
        #init display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        #init game state
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE,self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
        self.score = 0
        self.reward = 0
        self.food = None
        self._place__food()
        self.numberEmptyMoves = 0

    def _place__food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if(self.food in self.snake):
            self._place__food()


    def play_step(self,action):

        # Reset rewards from the previous step
        self.reward = 0

        # Reward for living
        self.reward += 0.0
        
        # Calculate distance to food
        distance_to_food_old = (self.food.x-self.head.x)**2 + \
                            (self.food.y-self.head.y)**2
        

        # 1. Collect the user input
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            
        # 2. Move
        turn_penalty = 0 # penalty for turning too much
        self._move(action,turn_penalty)
        self.snake.insert(0,self.head)

        # Calculate new distance to food and if smaller -> award
        distance_to_food_new = (self.food.x-self.head.x)**2 + \
                            (self.food.y-self.head.y)**2
        
        if distance_to_food_new<distance_to_food_old:
            self.reward +=0.0
        else:
            self.reward -=0.0


        # 3. Check if game Over
        game_over = False        
        if (self.is_collision() or self.numberEmptyMoves>100*len(self.snake)):
            game_over=True
            self.reward -= 10
            return self.reward,game_over,self.score
        

        # 4. Place new Food or just move
        if(self.head == self.food):
            self.score +=1
            self.reward +=10
            self.numberEmptyMoves = 0
            self._place__food()
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return game Over and Display Score
        return self.reward,game_over,self.score

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display,BLUE1,pygame.Rect(
                                    pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(self.display,BLUE2,pygame.Rect(
                                    pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display,RED,pygame.Rect(
                                self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render("Score: "+str(self.score),True,WHITE)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def _move(self,action,turn_penalty):
        # Action
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn 
        # [0,0,1] -> Left Turn

        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
            self.reward -= turn_penalty
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn
            self.reward -= turn_penalty

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if(self.direction == Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(self.direction == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(self.direction == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(self.direction == Direction.UP):
            y-=BLOCK_SIZE
        self.head = Point(x,y)

    def is_collision(self,pt=None):
        if(pt is None):
            pt = self.head
        #hit boundary
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.snake[1:]):
            return True
        return False