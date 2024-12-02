from typing import List
import random
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
#import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
#from torch_geometric.nn import GCNConv, SAGEConv

from CONST import (
    BUF_CAPASITY, EPSILON, MINEPSILON, TRAIN_EPS, MAX_TRAIN_EPS, NEXTEPSILON_PARAMETER, GAMMA, LEARNINGRATE,
    DOUBLEMODE, DUELINGMODE,NOISYMODE,LINEAREPSMODE,EACH_ROBOT_ACTION_MODE,
    NEXTEPSILONS_GAMMA,
    FULLNODEDATAMODE,ALLPARTIALMODE,ALLFULLNODEDATAMODE,SINGLE_AGENT_MODE,SINGLEGCNMODE,
    DEVICE,
    KNN_AGENTS, KNN_MODE
    )
from QNet import QNet

class DeepQPolicy:
    def __init__(self, state_dim, action_dim, graph):
        self.loss = 0 # もしミニバッチがうまる前にlossを要求されたらこれが呼ばれる
        self.device = DEVICE
        #self.num_agents = num_agents
        self.action_dim = action_dim
        self.graph = graph
        if SINGLE_AGENT_MODE:
            self.single_act_dim = self.action_dim
        #else:
        #    self.single_act_dim = self.action_dim // self.num_agents

        if SINGLE_AGENT_MODE:
            self._q_network = QNet(state_dim, action_dim).to(device=self.device)
            self._target_network = QNet(state_dim, action_dim).to(device=self.device)
            self.optimizer = optim.Adam(self._q_network.parameters(), lr=LEARNINGRATE)
        else:
            self._q_network = QNet(state_dim, action_dim).to(device=self.device)
            self._target_network = QNet(state_dim, action_dim).to(device=self.device)
            self.optimizer = optim.Adam(self._q_network.parameters(), lr=LEARNINGRATE)
        
        

        self.sync()
        
        self._q_network.eval()
        self._target_network.eval()
        
        self._epsilon = EPSILON
        self._minepsilon = MINEPSILON #add sugimoto
    
    def set_num_agents(self, num_agents):
        self.num_agents = num_agents
    
    def reset_env(self, actionlist, connect_dict, connect_to_dict):
        self._q_network.reset_env(actionlist, connect_dict, connect_to_dict)
        self._target_network.reset_env(actionlist, connect_dict, connect_to_dict)


    def load_network(self, trainfile, targetfile):
        
        self._q_network = torch.load(trainfile)
        self._target_network = torch.load(targetfile)
        self._q_network.eval()
        self._target_network.eval()
        
        #summary(model=self._q_network)

    def save_network(self, trainfile, targetfile):
        torch.save(self._q_network, trainfile)
        torch.save(self._target_network, targetfile)
    
    def save_network_param(self, trainfile, targetfile):
        torch.save(self._q_network.state_dict(), trainfile)
        torch.save(self._target_network.state_dict(), targetfile)

    def sync(self):
        self._target_network.load_state_dict(self._q_network.state_dict())
    
    #@profile
    def single_state_maker(self, state, idx):
        if KNN_MODE:
            data = state[self.graph.knn_agents[idx]]
            # 案1 self.knn_agents_distanseをそのままdataにここでくっつける。次元の変更が必要。
            return np.concatenate(data)
        else:
            data = np.insert(np.delete(state, idx, axis=0),0, state[idx], axis=0)
            return np.concatenate(data)
        #return a#np.insert(np.delete(state, idx, axis=0),0, state[idx], axis=0)
    
    #@profile #GNN 23% of savetraining
    #def abst_greedy(self, network, state, connectdict):
    def abst_greedy(self, network, state, info):
        if SINGLE_AGENT_MODE:
            #'''
            return self.single_greedy(network, state, info)
            #'''
            outputs_ = []
            state = state.reshape(self.num_agents, -1)
            for i in range(self.num_agents):
                an_input = self.single_state_maker(state, i)
                an_input = an_input.reshape(-1, len(an_input))
                an_input = torch.tensor(an_input).to(device=self.device)
                output = network(an_input, [connectdict])
                outputs_.append(torch.argmax(output).to(device="cpu").numpy())
            return np.array(outputs_)
        else:
            if state.ndim == 1:
                state = np.array([state])
            input_tensor_ = torch.tensor(state).to(device=self.device)
            output_ = network(input_tensor_) # GCN_Time 88.6%
            action_output_split = torch.reshape(output_[0],(-1, self.single_act_dim))
            return action_output_split.argmax(dim=1)
    
    def single_greedy(self, network, state, info):
        local_edge, rows_common, each_info = info
        inputs = state.reshape(self.num_agents, -1)
        inputs = torch.tensor(inputs).to(device=self.device)
        each_info = torch.tensor(each_info).to(device=self.device)
            
        outputs_ = network(inputs, local_edge, rows_common, each_info) # 97.1% is this process 
        return torch.argmax(outputs_, dim=1).to(device="cpu").numpy()
        
    
    def target_greedy(self, state, info):
        return self.abst_greedy(self._target_network, state, info)
    
    def greedy(self, state, info):
        return self.abst_greedy(self._q_network, state, info)
    
    def epsilon_greedy(self, state, info):
        ###
        if random.random() < self._epsilon:
            return np.array([random.randrange(0,self.single_act_dim) for i in range(self.num_agents)])
        else:
            return self.greedy(state, info)
    
    #@profile
    def train_batch(self, batch):
        encoded_s, action, reward, encoded_new_s, done, local_edge , rows_common, each_info, local_edge_next , rows_common_next, each_info_next= batch
        
        #encoded_s, action, reward, encoded_new_s, done, connectdict = batch

        td_target = self._q_network(encoded_s, local_edge, rows_common, each_info)
        q = self._q_network(encoded_s, local_edge, rows_common, each_info)
        
        td_target_new = self._q_network(encoded_new_s, local_edge_next, rows_common_next, each_info_next).detach().cpu().numpy()
        max_estimated_target = self._target_network(encoded_new_s, local_edge_next, rows_common_next, each_info_next).detach()
        
        
        for i in range(len(done)):
            t_a = np.argmax(td_target_new[i])
            renew_target = reward[i] + (1-int(done[i]))*GAMMA*max_estimated_target[i][t_a]
            td_target[i, action[i]] = renew_target  
        
        self._q_network.train()
        loss_fn = nn.MSELoss()
        loss = loss_fn(q, td_target)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward() # Time 22.8%
        self.optimizer.step()
        self._q_network.eval()

    def make_action_idxes(self, action):
        newaction = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            robot_offset = i * self.single_act_dim
            newaction[i] = robot_offset + action[i]
        return newaction
    
    def first_set_epsilon(self, numofeps):
        if LINEAREPSMODE:
            self._epsilon = max(self._epsilon-numofeps*NEXTEPSILON_PARAMETER, self._minepsilon)
        else:
            self._epsilon = max(self._epsilon*(NEXTEPSILONS_GAMMA**numofeps), self._minepsilon)
            
    def setepsilon(self):
        self._epsilon = max(self.nextepsilon(self._epsilon), self._minepsilon)
    
    def nextepsilon(self, epsilon):
        if LINEAREPSMODE:
            return epsilon - NEXTEPSILON_PARAMETER
        else:
            return epsilon * NEXTEPSILONS_GAMMA
