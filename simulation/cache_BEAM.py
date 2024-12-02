import numpy as np
from MultiRobotsEnvironment import MultiRobotsEnvironment
import heapq
from collections import defaultdict


class BeamPolicy:
    def __init__(self, env:MultiRobotsEnvironment, beam_width=10):
        self.env = env
        self.zeromoves = np.zeros(self.env.num_agents, dtype=np.int32)
        self.moves = np.zeros(self.env.num_agents, dtype=np.int32)
        self.beam = CBSBeamSearch(self.env.connect_dict, self.env.num_agents, self.env.node_distances)
        self.cache_path = None
    
    def greedy(self, curr_state, dest_state):
        self.moves[:] = self.zeromoves[:]
        
        refing_flag = False
        if self.cache_path == None:
            refing_flag = True
        else:
            for agent in range(self.env.num_agents):
                if len(self.cache_path[agent]) < 2: # change 1->2
                    refing_flag = True
        
        if refing_flag:
            cbs_path = self.beam.find_paths(curr_state, dest_state)
        else:
            cbs_path = self.cache_path
        
        for agent in range(self.env.num_agents):
            actions = self.env.actionlist[self.env.robot_envs[agent].state_node]
            for action_cand in range(len(actions)):
                if len(cbs_path[agent]) == 1:
                    self.moves[agent] = 0
                elif cbs_path[agent][1] == actions[action_cand]:
                    self.moves[agent] = action_cand
            
            _ = cbs_path[agent].pop(0)
                
        self.cache_path = cbs_path
        
        return self.moves

class CBSBeamSearch:
    def __init__(self, graph, num_agents, distances, beam_width=5) -> None:
        self.graph = graph
        self.num_agents = num_agents
        self.distances = distances
        self.beam_width = beam_width
        
    def find_paths(self, curr_state, dest_state):
        open_list = []
        root = {
            'cost':0,
            'constraints':[],
            'paths':[self.a_star(agent, curr_state, dest_state) for agent in range(self.num_agents)]
        }
        priority = 0
        heapq.heappush(open_list, (root['cost'], priority, root))
        cnt = 0
        
        gabage = []
        
        while open_list and cnt < 10 * self.num_agents:
            new_open_list = []
            
            while open_list and len(new_open_list) < self.beam_width:
                cnt += 1
                _, _, node = heapq.heappop(open_list)
                gabage = node['paths']
                
                conflict = self.get_first_conflict(node['paths'])
                if not conflict:
                    return node['paths']
                
                constraints = self.create_constraints(conflict)
                for constraint in constraints:
                    new_node = self.create_node(node, curr_state, dest_state, constraint)
                    if new_node:
                        new_cost = new_node['cost']
                        priority += 1
                        heapq.heappush(new_open_list, (new_cost, priority, new_node))
            
            open_list = new_open_list

        if open_list:
            _, _, node = heapq.heappop(open_list)
            return node['paths']
        else:
            return gabage
    
    def a_star(self, agent, curr_state, dest_state, constraints=None):
        start = curr_state[agent]
        goal = dest_state[agent]
        
        open_list = [(0, start)]
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for neighbor in self.graph[current]:
                if self.is_constrained(current, neighbor, g_score[current] + 1, constraints, agent):
                    continue
                
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        return None
    
    def heuristic(self, node_a, node_b):
        return self.distances[node_a, node_b]
    
    def reconstruct_path(self, came_from, current):
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]
    
    def get_first_conflict(self, paths):
        max_len = max(len(path) for path in paths)
        for t in range(max_len):
            positions = defaultdict(list)
            for agent, path in enumerate(paths):
                if t < len(path):
                    pos = path[t]
                else:
                    pos = path[-1]
                positions[pos].append(agent)
                if len(positions[pos]) > 1:
                    return {'time': t, 'pos': pos, 'agents': positions[pos]}
        return None

    def create_constraints(self, conflict):
        constraints = []
        for agent in conflict['agents']:
            constraints.append({
                'agent': agent,
                'pos': conflict['pos'],
                'time': conflict['time']
            })
        return constraints

    def create_node(self, node, curr_state, dest_state, constraint):
        new_constraints = node['constraints'] + [constraint]
        new_paths = []
        for agent in range(self.num_agents):
            new_path = self.a_star(agent, curr_state, dest_state, new_constraints)
            if not new_path:
                return None
            new_paths.append(new_path)
        return {
            'cost': sum(len(path) for path in new_paths),
            'constraints': new_constraints,
            'paths': new_paths
        }

    def is_constrained(self, current, next_pos, time, constraints, agent):
        if constraints is None:
            return False
        for constraint in constraints:
            if constraint['pos'] == next_pos and constraint['time'] == time and constraint['agent'] == agent:
                return True
        return False
