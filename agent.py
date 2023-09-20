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

# How many steps we train on
BATCH_SIZE = 1000

# #How many steps to update the target network and batch-train the model
# #TARGET_UPDATE_FREQ = 100

FRESHSTART = False

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0.5 # Proportion of random moves initially
        self.N_exploration_games = 0
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(12,288,36,3) 
        if not FRESHSTART:
            modelpath = r'C:\Users\Asus\Documents\Coding\Python\Machine Learning\SnakeGameRL\model\model.pth'
            self.model.load_state_dict(torch.load(modelpath))

        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)


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

        #distance_to_food = (game.food.x-head.x)**2 + (game.food.y-head.y)**2

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #Danger Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l))or
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

            # Length of the snake
            len(game.snake)
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_network_on_step(self, state, action, reward, next_state, done):
        self.trainer.train_model(
                            state,action,reward,next_state,done)
        
    def update_target_network_and_train(self):
        """
        Train the model network and
        update the target network with current model weights
        """
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_model(
                            states,actions,rewards,next_states,dones)
        self.trainer.update_targetNN()
        return loss
    
    def train_network(self):
        """
        Train the model network based on memory (s,a)
        """
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_model(states,actions,rewards,next_states,dones)

    def get_action(self,state):
        # random moves: tradeoff explotation / exploitation
        final_move = [0,0,0]

        if (self.n_game<self.N_exploration_games and 
                                self.epsilon*self.N_exploration_games>0):
            if(random.randint(0, int(self.N_exploration_games)) < 
                        self.epsilon*self.N_exploration_games):
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
    
    # def get_action(self, state):
    #     # random moves: tradeoff exploration / exploitation
    #     self.epsilon = 80 - self.n_game
    #     final_move = [0,0,0]
    #     if random.randint(0, 200) < self.epsilon:
    #         move = random.randint(0, 2)
    #         final_move[move] = 1
    #     else:
    #         state0 = torch.tensor(state, dtype=torch.float).cuda()
    #         prediction = self.model(state0).cuda()
    #         move = torch.argmax(prediction).item()
    #         final_move[move] = 1

    #     return final_move

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

        # Train on the step taken
        agent.train_network_on_step(state_old, final_move,
                                    reward, state_new, done)

        # Store information about states and actions into memory        
        agent.remember(state_old,final_move,reward,state_new,done)

        # Every N steps train on the larger batch and update target network
        # if target_update_iter > TARGET_UPDATE_FREQ:
        #     #agent.update_target_network_and_train() 
        #     target_update_iter = 0

        if done:
            # Iterable for updating the explore/exploitation ratio of moves
            #agent.N_exploration_games -= agent.epsilon

            # Update target network
            game.reset()
            agent.n_game += 1
            
            # Find loss for plotting
            loss = agent.update_target_network_and_train() 
            
             # Save the best model
            if(score > record): 
                record = score
                agent.model.save()

            print('Game:',agent.n_game,'Score:',score, 
                    "Record:", record)
            
           
            plot_scores.append(score)
            losses.append(loss)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores, losses)

        train_model_iter +=1
        target_update_iter +=1

if(__name__=="__main__"):
    train()