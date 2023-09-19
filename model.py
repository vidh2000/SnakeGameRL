import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from copy import deepcopy

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)#.cuda()
        #self.relu1 = nn.ReLU()#.cuda()
        #self.linear2 = nn.Linear(hidden_size1,hidden_size2)#.cuda()
        #self.relu2 = nn.ReLU()#.cuda()
        self.linear3 = nn.Linear(hidden_size,output_size)#.cuda()
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = self.relu1(x)
        #x = self.linear2(x)
        #x = self.relu2(x)
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = r'C:\Users\Asus\Documents\Coding\Python\Machine Learning\SnakeGameRL\model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimiser = optim.Adam(model.parameters(),lr = self.lr)    
        self.criterion = nn.MSELoss() #nn.HuberLoss()
        
    # def update_targetNN(self):
    #     # Create a copy of the DNN with the same weights
    #     self.targetNN = deepcopy(self.model)

        
    def train_model(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)#.cuda()
        next_state = torch.tensor(next_state,dtype=torch.float)#.cuda()
        action = torch.tensor(action,dtype=torch.long)#.cuda()
        reward = torch.tensor(reward,dtype=torch.float)#.cuda()

        # only one parameter to train in short-memory
        # Hence convert to tuple of shape (1, x)
        if(len(state.shape) == 1):
            state = torch.unsqueeze(state,0)#.cuda()
            next_state = torch.unsqueeze(next_state,0)#.cuda()
            action = torch.unsqueeze(action,0)#.cuda()
            reward = torch.unsqueeze(reward,0)#.cuda()
            done = (done, )
        
        # Get comparison tensor of Q for training
        pred = self.model(state)#.cuda()
        y = pred.clone()#.cuda()

        #######################################################################
        # Q-values from model network updated according to Bellman equation
        # and compared with old Q values to perform gradient descent to update
        # model network parameters
        # Q == output weight of the last nodes in DNN
        #######################################################################

        # Iterate through all states + decisions in the past and 
        # update y according to Q' obtained from the target network

        
        for t in range(len(done)):
            Q_new = reward[t]
            if not done[t]:
                Q_new = reward[t] + self.gamma * torch.max(self.model(next_state[t]))#.cuda()
            y[t][torch.argmax(action[t]).item()] = Q_new 

        # Train the model
        self.optimiser.zero_grad()
        loss = self.criterion(y,pred)
        loss_float = loss.item()
        loss.backward()

        self.optimiser.step()
        return loss_float
  