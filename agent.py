import torch 
import random 
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
from Helper import plot
import multiprocessing as mp

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
TRAIN_FREQ = 25

FRESHSTART = True

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0.9 # Proportion of random moves initially
        self.N_eps_steps = 1000
        self.gamma = 0.9 # discount rate
        self.alpha =0.1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,33,33,3) 
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

    #   --------------------- state (11 Values) ------------------------
    # [
    #   danger straight, danger right, danger left, 
    #   
    #   direction left, direction right,
    #   direction up, direction down
    # 
    #   food left,food right,
    #   food up, food down
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
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        loss = self.trainer.train_step(
                            states,actions,rewards,next_states,dones)
        return loss
    
    # def train_short_memory(self,state,action,reward,next_state,done,
    #                                 short_mem_train_freq):
    #     # # For single move training
    #     # self.trainer.train_step(state,action,reward,next_state,done)
    #     if (len(self.memory) > short_mem_train_freq):
    #         mini_sample = random.sample(self.memory,short_mem_train_freq)
    #     else:
    #         mini_sample = self.memory
    #     states,actions,rewards,next_states,dones = zip(*mini_sample)
    #     self.trainer.train_step(states,actions,rewards,next_states,dones)

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
    plot_scores = []
    plot_mean_scores = []
    losses = []
    loss = 0
    short_memory_iter = 0
    total_score = 0
    record = 0
    
    agent = Agent()
    

    game = SnakeGameAI()
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

        if done:
            # Iterable for how often to train and for explor/exploit
            short_memory_iter +=1
            agent.N_eps_steps -= agent.epsilon
            # Train long memory,plot result
            game.reset()
            agent.n_game += 1
            if short_memory_iter > TRAIN_FREQ:
                short_memory_iter = 0
                loss = agent.train_long_memory()

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

# Work in progress for faster training..
def trainInParallel(N):
  
    fns = []
    for i in range(N):
        fns.append(train)
    proc = []
    for fn in fns:
        p = mp.Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

if(__name__=="__main__"):

    train()
    #trainInParallel(2)