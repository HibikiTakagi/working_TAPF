import random
import numpy as np
from CONST import BUF_CAPASITY, DEVICE
import torch
import copy

"""
class ReplayBuffer:
    def __init__(self, capacity=BUF_CAPASITY):
        self._buffer = []
        self._capacity = capacity
        self._next_idx = 0

    def __len__(self):
        return len(self._buffer)

    def is_full(self):
        return len(self._buffer) == self._capacity

    def add(self, transition):
        if self._next_idx >= len(self._buffer):
            self._buffer.append(transition)
        else:
            self._buffer[self._next_idx] = transition

        self._next_idx = (self._next_idx + 1) % self._capacity

    def sample(self, batch_size: int):
        indexes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        
        return [self._buffer[i] for i in indexes]
"""

class ReplayBuffer:
    """Simple Replay Buffer implementation"""

    def __init__(self, capacity=BUF_CAPASITY):
        self._capacity = capacity
        self._next_idx = 0
        self._state_buffer = None
        self._action_buffer = None
        self._reward_buffer = None
        self._next_state_buffer = None
        self._done_buffer = None
        self.volume = 0
        self.is_empty = True

    def __len__(self):
        #return len(self._state_buffer)
        return self.volume

    def is_full(self):
        return len(self.volume) == self._capacity

    def add(self, transition):
        #state, action, reward, next_state, done = transition
        #state, action, reward, next_state, done, local_edge, rows_common, rows_local, nodes_info_for_gnn, each_info,local_edge_next, rows_common_next, rows_local_next, nodes_info_for_gnn_next, each_info_next = transition
        #state, action, reward, next_state, done, local_edge, rows_common, rows_local, each_info,local_edge_next, rows_common_next, rows_local_next, each_info_next = transition

        state, action, reward, next_state, done, local_edge, rows_common, each_info,local_edge_next, rows_common_next, each_info_next = transition

        if self._next_idx==0 and self.is_empty:
            self._state_buffer = torch.tensor(np.array([state for _ in range(self._capacity)]), requires_grad=False).to(device=DEVICE)
            self._action_buffer = np.array([action for _ in range(self._capacity)])
            self._reward_buffer = np.array([reward for _ in range(self._capacity)])
            self._next_state_buffer = torch.tensor(np.array([next_state for _ in range(self._capacity)]), requires_grad=False).to(device=DEVICE)
            self._done_buffer = np.array([done for _ in range(self._capacity)])
            
            self._each_info_dict_buffer = torch.tensor(np.array([each_info for _ in range(self._capacity)]), requires_grad=False).to(device=DEVICE)
            self._each_info_next_dict_buffer = torch.tensor(np.array([each_info_next for _ in range(self._capacity)]), requires_grad=False).to(device=DEVICE)
        else:
            self._state_buffer[self._next_idx] = torch.tensor(state, requires_grad=False).to(device=DEVICE)
            #self._action_buffer[self._next_idx] = action[:]
            self._action_buffer[self._next_idx] = action
            self._reward_buffer[self._next_idx] = reward
            self._next_state_buffer[self._next_idx] = torch.tensor(next_state, requires_grad=False).to(device=DEVICE)
            self._done_buffer[self._next_idx] = done
            
            self._each_info_dict_buffer[self._next_idx] = torch.tensor(each_info, requires_grad=False).to(device=DEVICE)
            self._each_info_next_dict_buffer[self._next_idx] = torch.tensor(each_info_next, requires_grad=False).to(device=DEVICE)
        
        if self._next_idx==0 and self.is_empty:
            #self._connect_dict_buffer = [copy.deepcopy(connectdict)]
            self._local_edge_dict_buffer = [copy.deepcopy(local_edge)]
            self._rows_common_dict_buffer = [copy.deepcopy(rows_common)]
            #self._rows_local_dict_buffer = [copy.deepcopy(rows_local)]
            #self._nodes_info_for_gnn_dict_buffer = [copy.deepcopy(nodes_info_for_gnn)]
            #self._each_info_dict_buffer = [copy.deepcopy(each_info)]
            #self._each_info_dict_buffer = torch.tensor(np.array([each_info for _ in range(self._capacity)]), requires_grad=False).to(device=DEVICE)
            
            
            self._local_edge_next_dict_buffer = [copy.deepcopy(local_edge_next)]
            self._rows_common_next_dict_buffer = [copy.deepcopy(rows_common_next)]
            #self._rows_local_next_dict_buffer = [copy.deepcopy(rows_local_next)]
            #self._nodes_info_for_gnn_next_dict_buffer = [copy.deepcopy(nodes_info_for_gnn_next)]
            #self._each_info_next_dict_buffer = [copy.deepcopy(each_info_next)]
            #self._each_info_next_dict_buffer = torch.tensor(np.array([each_info_next for _ in range(self._capacity)]), requires_grad=False).to(device=DEVICE)
            
        elif self.volume == self._capacity:
            #self._connect_dict_buffer[self._next_idx] = copy.deepcopy(connectdict)
            self._local_edge_dict_buffer[self._next_idx] = copy.deepcopy(local_edge)
            self._rows_common_dict_buffer[self._next_idx] = copy.deepcopy(rows_common)
            #self._rows_local_dict_buffer[self._next_idx] = copy.deepcopy(rows_local)
            #self._nodes_info_for_gnn_dict_buffer[self._next_idx] = copy.deepcopy(nodes_info_for_gnn)
            #self._each_info_dict_buffer[self._next_idx] = copy.deepcopy(each_info)

            self._local_edge_next_dict_buffer[self._next_idx] = copy.deepcopy(local_edge_next)
            self._rows_common_next_dict_buffer[self._next_idx] = copy.deepcopy(rows_common_next)
            #self._rows_local_next_dict_buffer[self._next_idx] = copy.deepcopy(rows_local_next)
            #self._nodes_info_for_gnn_next_dict_buffer[self._next_idx] = copy.deepcopy(nodes_info_for_gnn_next)
            #self._each_info_next_dict_buffer[self._next_idx] = copy.deepcopy(each_info_next)
        else:
            #self._connect_dict_buffer.append(copy.deepcopy(connectdict))
            self._local_edge_dict_buffer.append(copy.deepcopy(local_edge))
            self._rows_common_dict_buffer.append(copy.deepcopy(rows_common))
            #self._rows_local_dict_buffer.append(copy.deepcopy(rows_local))
            #self._nodes_info_for_gnn_dict_buffer.append(copy.deepcopy(nodes_info_for_gnn))
            #self._each_info_dict_buffer.append(copy.deepcopy(each_info))

            self._local_edge_next_dict_buffer.append(copy.deepcopy(local_edge_next))
            self._rows_common_next_dict_buffer.append(copy.deepcopy(rows_common_next))
            #self._rows_local_next_dict_buffer.append(copy.deepcopy(rows_local_next))
            #self._nodes_info_for_gnn_next_dict_buffer.append(copy.deepcopy(nodes_info_for_gnn_next))
            #self._each_info_next_dict_buffer.append(copy.deepcopy(each_info_next))
            

        self.is_empty = False
        self._next_idx = (self._next_idx + 1) % self._capacity
        self.volume = min(self.volume+1,self._capacity)

    def sample(self, batch_size: int):
        indexes = [random.randint(0, self.volume - 1) for _ in range(batch_size)]
        #states = torch.tensor(np.array([self._state_buffer[i] for i in indexes]), requires_grad=True).to(device=DEVICE)
        #actions = np.array([self._action_buffer[i] for i in indexes])
        #rewards = np.array([self._reward_buffer[i] for i in indexes])
        #next_states = torch.tensor(np.array([self._next_state_buffer[i] for i in indexes]), requires_grad=True).to(device=DEVICE)
        #done = np.array([self._done_buffer[i] for i in indexes])

        states = self._state_buffer[indexes]
        actions = self._action_buffer[indexes]
        rewards = self._reward_buffer[indexes]
        next_states = self._next_state_buffer[indexes]
        done = self._done_buffer[indexes]
        
        each_info = self._each_info_dict_buffer[indexes]
        each_info_next = self._each_info_next_dict_buffer[indexes]
        
        #connectdict = [self._connect_dict_buffer[ind] for ind in indexes]
        local_edge = [self._local_edge_dict_buffer[ind] for ind in indexes]
        rows_common = [self._rows_common_dict_buffer[ind] for ind in indexes]
        #rows_local = [self._rows_local_dict_buffer[ind] for ind in indexes]
        #nodes_info_for_gnn = [self._nodes_info_for_gnn_dict_buffer[ind] for ind in indexes]
        #each_info = [self._each_info_dict_buffer[ind] for ind in indexes]
        
        local_edge_next = [self._local_edge_next_dict_buffer[ind] for ind in indexes]
        rows_common_next = [self._rows_common_next_dict_buffer[ind] for ind in indexes]
        #rows_local_next = [self._rows_local_next_dict_buffer[ind] for ind in indexes]
        #nodes_info_for_gnn_next = [self._nodes_info_for_gnn_next_dict_buffer[ind] for ind in indexes]
        #each_info_next = [self._each_info_next_dict_buffer[ind] for ind in indexes]
        
        return [states, actions, rewards, next_states, done, local_edge, rows_common, each_info, local_edge_next, rows_common_next, each_info_next]
        
        #return [states, actions, rewards, next_states, done, local_edge, rows_common, rows_local, nodes_info_for_gnn, each_info,local_edge_next, rows_common_next, rows_local_next, nodes_info_for_gnn_next, each_info_next]
        #return [states, actions, rewards, next_states, done, local_edge, rows_common, rows_local, each_info,local_edge_next, rows_common_next, rows_local_next, each_info_next]
        #return [states, actions, rewards, next_states, done]