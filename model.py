import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1).cuda()
        #self.relu1 = nn.ReLU().cuda()
        self.linear2 = nn.Linear(hidden_size1,hidden_size2).cuda()
        #self.relu2 = nn.ReLU().cuda()
        self.linear3 = nn.Linear(hidden_size2,output_size).cuda()
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = self.relu1(x)
        x = self.linear2(x)
        #x = self.relu2(x)
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = r'C:\Users\Asus\Documents\Coding\Python\Machine Learning\SnakeGameRL' 
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimiser = optim.Adam(model.parameters(),lr = self.lr)    
        self.criterion = nn.HuberLoss() #nn.MSELoss()
        # for i in self.model.parameters():
        #     print(i.is_cuda)

    
    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float).cuda()
        next_state = torch.tensor(next_state,dtype=torch.float).cuda()
        action = torch.tensor(action,dtype=torch.long).cuda()
        reward = torch.tensor(reward,dtype=torch.float).cuda()

        # only one parameter to train in short-memory
        # Hence convert to tuple of shape (1, x)
        if(len(state.shape) == 1):
            state = torch.unsqueeze(state,0).cuda()
            next_state = torch.unsqueeze(next_state,0).cuda()
            action = torch.unsqueeze(action,0).cuda()
            reward = torch.unsqueeze(reward,0).cuda()
            done = (done, )
        
        ##############################################################
        # Q_new updated according to Bellman equation
        # Q == output weight of the last nodes in DNN
        ##############################################################
        
        # Create a copy of the DNN to update Q-values
        pred = self.model(state).cuda()
        target = pred.clone().cuda()

        # Iterate through all states+decisions in the past and 
        # update weights according to Bellman equation
        for idx in range(len(done)):
            Q_old = pred[idx]
            Q_new = self.lr*reward[idx]
            if not done[idx]:
                Q_new =  reward[idx] + self.gamma * \
                        torch.max(self.model(next_state[idx])).cuda()
            target[idx][torch.argmax(action).item()] = Q_new 

        self.optimiser.zero_grad()
        loss = self.criterion(target,pred)
        loss_float = loss.item()
        loss.backward()

        self.optimiser.step()
        return loss_float
  
