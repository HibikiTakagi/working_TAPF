import numpy as np
import random
from MultiRobotsEnvironment import MultiRobotsEnvironment

class AnnealingPolicy:
    def __init__(self, env:MultiRobotsEnvironment, initial_temp=100, cooling_rate=0.99, min_temp=0.1):
        self.env = env
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_temp = initial_temp
    
    def set_num_agents(self, num_agents):
        self.num_agents = num_agents

    #def greedy(self, state):
        # Implement a greedy method if needed, otherwise remove it
    #    pass

    def anneal(self, current_cost, neighbor_cost):
        if neighbor_cost < current_cost:
            return True
        else:
            probability = np.exp((current_cost - neighbor_cost) / self.current_temp)
            return random.random() < probability

    def get_neighbors(self, state):
        # Implement logic to get neighboring states (possible next actions)
        neighbors = []
        for action in range(self.env.action_space.n):
            neighbors.append(action)
        return neighbors
    """
    def greedy(self, state):
        current_cost = self.env.get_cost(state)
        neighbors = self.get_neighbors(state)
        best_action = None
        best_cost = float('inf')

        for neighbor in neighbors:
            next_state, rw, _, _, _ = self.env.step(neighbor)
            neighbor_cost = self.env.get_cost(next_state)
            if self.anneal(current_cost, neighbor_cost):
                current_cost = neighbor_cost
                best_action = neighbor
                best_cost = neighbor_cost
            self.env.reset_to_state(state)  # Reset to original state

        self.current_temp = max(self.current_temp * self.cooling_rate, self.min_temp)
        return best_action if best_action is not None else random.choice(neighbors)
    """
    def greedy(self, state):
        inputs = state.reshape(self.num_agents, -1)
        outputs_ = np.array([self.value(inputs[idx], idx) for idx in range(self.num_agents)])
        return np.argmax(outputs_, axis=1)
    
    def value(self, state, idx):
        current_cost =  - self.env._reward_function_id(idx, self.env.cnt_all_privent_collision, 0, 0)
        best_action = None
        best_cost = float('inf')
        
        for action in range(self.env.net_act_dim):
            next_state, rw, _, _, _ = self.env.virtual_step(action)
            neighbor_cost = - rw
            if self.anneal(current_cost, neighbor_cost):
                current_cost = neighbor_cost
                best_action = action
                best_cost = neighbor_cost
            
        self.current_temp = max(self.current_temp * self.cooling_rate, self.min_temp)
        return best_action if best_action is not None else random.randrange(self.env.net_act_dim)#random.choice(actionlist)
            
                
            
        
        
        