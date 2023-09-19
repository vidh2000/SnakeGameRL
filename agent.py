import torch 
import random 
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
from Helper import plot
import multiprocessing as mp

MAX_MEMORY = 100_000
LR = 0.001

BATCH_SIZE = 1000 #how many steps we train on
TRAIN_FREQ = 1000 #how many steps we update params of our network
TARGET_UPDATE_FREQ = 10 #number of games to update the target network

FRESHSTART = True

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0.8 # Proportion of random moves initially
        self.N_eps_steps = 100
        self.gamma = 0.9 # discount rate
        self.alpha =0.1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(12,256,36,3) 
        if not FRESHSTART:
            modelpath = r'C:\Users\Asus\Documents\Coding\Python\Machine Learning\SnakeGameRL\model.pth'
            self.model.load_state_dict(torch.load(modelpath))

        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma,
                                alpha=self.alpha)
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)     

    #   --------------------- state (12 Values) ------------------------
    # [
    #   danger straight, danger right, danger left, 
    #   
    #   direction left, direction right,
    #   direction up, direction down
    # 
    #   food left,food right,
    #   food up, food down
    # 
    #   distance between the head and the apple
    # ]

    def get_state(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        distance_to_food = (game.food.x-head.x)**2 + (game.food.y-head.y)**2

        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger Left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, # food is in left
            game.food.x > game.head.x, # food is in right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y,  # food is down

            # Distance to food
            distance_to_food
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def update_target_network(self):
        """
        Train the model network and
        update the target network with current model weights
        """

        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        loss = self.trainer.train_model(
                            states,actions,rewards,next_states,dones)
        self.trainer.update_targetNN()
        return loss
    
    def train_network(self):
        """
        Train the model network based on memory (s,a)
        """
        if (len(self.memory) > TRAIN_FREQ):
            mini_sample = random.sample(self.memory,TRAIN_FREQ)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_model(states,actions,rewards,next_states,dones)

    def get_action(self,state):
        # random moves: tradeoff explotation / exploitation
        # for the first 100 games
        final_move = [0,0,0]

        if (self.n_game<self.N_eps_steps and self.epsilon*self.N_eps_steps>0):
            if(random.randint(0, int(self.N_eps_steps)) < self.epsilon*self.N_eps_steps):
                move = random.randint(0,2)
                final_move[move]=1

            else:
                state0 = torch.tensor(state,dtype=torch.float).cuda()
                prediction = self.model(state0).cuda() # prediction by model 
                move = torch.argmax(prediction).item()
                final_move[move]=1 
        else:
            state0 = torch.tensor(state,dtype=torch.float).cuda()
            prediction = self.model(state0).cuda() # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

def train():
    total_score = 0
    plot_scores = []
    plot_mean_scores = []
    record = 0
    losses = []
    loss = 0
    target_update_iter = 0
    train_model_iter = 0
    
    
    # Initialise required classes
    agent = Agent()
    game = SnakeGameAI()

    # Get target network with initial weights to be updated later
    agent.trainer.update_targetNN()

    while True:

        # Get Old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        # Count number of moves before getting to an apple
        game.numberEmptyMoves +=1
        state_new = agent.get_state(game)

        
        agent.remember(state_old,final_move,reward,state_new,done)

        if train_model_iter> TRAIN_FREQ:

            agent.train_network()
            train_model_iter = 0

        if done:
            # Iterables for how often to update target network
            # and for updating the explore/exploitation ratio of moves
            target_update_iter +=1
            agent.N_eps_steps -= agent.epsilon

            # Update target network
            game.reset()
            agent.n_game += 1
            if target_update_iter > TARGET_UPDATE_FREQ:
                loss = agent.update_target_network() # could change to be updated after N number of steps, not games
                target_update_iter = 0

            print('Game:',agent.n_game,'Score:',score, 
                    "Record:", record, 
                    "Reward =", round(reward,3))
            if(score > record): # new High score 
                record = score
                agent.model.save()
            #print('Game:',agent.n_game,'Score:',score)
            
            plot_scores.append(score)
            losses.append(loss)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores, losses)

        train_model_iter +=1

if(__name__=="__main__"):

    train()