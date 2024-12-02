import numpy as np
from MultiRobotsEnvironment import MultiRobotsEnvironment
from heapq import heapify, heappush, heappop, heappushpop
#from map_maker import  NODE_DISTANCES, ACTION_LIST
"""
class BeamPolicy:
    def __init__(self, env:MultiRobotsEnvironment):
        self.env = env
        self.zeromoves = np.zeros(self.env.num_agents, dtype=np.int32)
        self.moves = np.zeros(self.env.num_agents, dtype=np.int32)
    def greedy(self, state):
        ###
        '''
        train greedy
        '''
        self.moves[:] = self.zeromoves[:]
        for i in range(self.env.num_agents):
            robot = self.env.robot_envs[i]
            if robot.state_node == robot.dest_node:
                dist = self.env.node_distances[robot.state_node, robot.tasks_node[min(robot.task_id+1, len(robot.tasks_node)-1)]]
            else:
                dist = self.env.node_distances[robot.state_node, robot.dest_node]
            nextnodes = self.env.actionlist[robot.state_node]

            action = 0
            for j in range(len(nextnodes)):
                nextnode = nextnodes[j]
                if robot.state_node == robot.dest_node:
                    tmpdist = self.env.node_distances[nextnode, robot.tasks_node[min(robot.task_id+1, len(robot.tasks_node)-1)]]
                else:
                    tmpdist = self.env.node_distances[nextnode, robot.dest_node]
                if dist > tmpdist:
                    action = j
            self.moves[i]=action

        return self.moves
"""
import heapq
import random
from MultiRobotsEnvironment import MultiRobotsEnvironment

class BeamPolicy:
    def __init__(self, env: MultiRobotsEnvironment, beam_width=3):
        self.env = env
        self.beam_width = beam_width
    
    def set_num_agents(self, num_agents):
        self.num_agents = num_agents

    """
    def get_neighbors(self, state):
        neighbors = []
        #for action in range(self.env.net_act_dim * self.num_agents):
        sample = 20
        for action in [np.random.randint(0, self.env.net_act_dim, self.num_agents) for _ in range(sample)]:
            
            #print(self.env.virtual_step(action))
            next_state, reward, done, _,_ = self.env.virtual_step(action)
            #print(reward)
            neighbors.append((next_state, reward, done, action))
        return neighbors

    def greedy(self, state):
        beam = [(0, state, [])]  # (cost, state, path)
        best_path = []
        cnt = 0
        #print(state)
        while beam and cnt < 3:
            cnt += 1
            next_beam = []
            for cost, current_state, path in beam:
                #if len(path) > 0 and self.env.is_goal_state(current_state):
                #    return path[0]  # Return the first action of the best path
                print("@@@@@@@@")
                print((cost, current_state, path))
                #print(f"heapq:{next_beam}")
                
                for i in range(len(next_beam)):
                    beam_cost, beam_state, beam_path = next_beam[i]
                    print(f"{i}:")
                    print(f"cost:{beam_cost}, state:{beam_state}, path:{beam_path}")
                
                
                neighbors = self.get_neighbors(current_state)
                for next_state, reward, done, action in neighbors:
                    next_cost = cost - reward  # Negative reward to minimize cost
                    next_path = path + [action]
                    #print("########")
                    #print( (next_cost, next_state, next_path))
                    
                    heapq.heappush(next_beam, (next_cost, next_state, next_path))

            beam = heapq.nsmallest(self.beam_width, next_beam)

        return beam[0][2][0] if beam else np.random.randint(0, self.env.net_act_dim, self.num_agents)  # Default to random action if no beam
        """
        
    def beam_search(root, k, api):
        """
        Args:
            root : root node
            k : number of remain paths during search
            api : apis for beam search
        Notes:
            api must have functions as follows.
            (1) init : this is called at the begenning of this function
            (2) step : return path-list or path-generator of extended path from inputed path
            (3) score : return score for path, higher scores indicate better
            (4) count : this function is called for every end of loop
            (5) terminate : return true if it should terminate to search else false
        """
        paths = [(None, root)]
        heapify(paths)
        api.init()
        while not api.terminate():
            top_paths = []
            heapify(top_paths)
            for _, path in paths:
                for extend_path in api.step(path):
                    score = api.score(extend_path)
                    if len(top_paths) < k:
                        heappush(top_paths, (score, extend_path))
                    else:
                        heappushpop(top_paths, (score, extend_path))
            paths = top_paths
            api.count()

        result_paths = []
        result_paths_score = []
        for _, path in paths:
            result_paths.append(path)
            result_paths_score.append(score)

        return result_paths, result_paths_score
    
    def greedy(self, state):
        return
