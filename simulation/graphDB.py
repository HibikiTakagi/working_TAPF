import numpy as np

from CONST import NUM_AGENTS, KNN_AGENTS, PARTIAL_AGENT_MODE, LOCALGCNMODE, KNN_REWARD_AGENTS, DISTANCE
from collections import deque
from map_maker import TASK_ASSIGN_MODE, SINGLE_TASK_MODE

NODE_DATA_COMMON_DIM = 3
#NODE_DATA_COMMON_DIM = 2
NODE_INFO_COMMON_START = 0
NODE_INFO_COMMON_NUMROBOT = 1

NODE_INFO_COMMON_PICKER_BOOL = None

NODE_INFO_COMMON_ROUTE = 2

POSMODE = False
if POSMODE:
    NODE_DATA_COMMON_DIM = 4
    NODE_INFO_COMMON_POSX = 2
    NODE_INFO_COMMON_POSY = 3


NODE_DATA_EACH_DIM = 2
#NODE_DATA_EACH_DIM = 3

NODE_DATA_DIM = NODE_DATA_EACH_DIM + NODE_DATA_COMMON_DIM
NODE_INFO_STATE = 0
NODE_INFO_ROUTE = 1

#NODE_INFO_BEFORE_STATE = 2
if NODE_INFO_COMMON_START is not None:
    NODE_INFO_A_START = NODE_DATA_EACH_DIM + NODE_INFO_COMMON_START
if NODE_INFO_COMMON_ROUTE is not None:
    NODE_INFO_A_ROUTE = NODE_DATA_EACH_DIM + NODE_INFO_COMMON_ROUTE
else:
    NODE_INFO_A_ROUTE = NODE_DATA_EACH_DIM
if NODE_INFO_COMMON_NUMROBOT is not None:
    NODE_INFO_A_NUMROBOT = NODE_DATA_EACH_DIM + NODE_INFO_COMMON_NUMROBOT

if POSMODE:
    NODE_INFO_A_POSX = NODE_DATA_EACH_DIM + NODE_INFO_COMMON_POSX
    NODE_INFO_A_POSY = NODE_DATA_EACH_DIM + NODE_INFO_COMMON_POSY



class Graph:
    def __init__(self, len_node, node_distances, connectdict, connect_to_dict, coord_list) -> None:
        self.reset(len_node, node_distances, connectdict, connect_to_dict, coord_list)
        
    
    def reset(self, lennode, node_distances, connect_dict, connect_to_dict, coord_list):
        self.len_node = lennode
        self.node_distances = node_distances
        self.connect_dict = connect_dict
        self.connect_to_dict = connect_to_dict
        self.position_dict = coord_list

        self.zeros_node = np.zeros(lennode, dtype=np.float32) # 初期化用　ただし使いたくない。
        self.zeros_bool_node = np.zeros(lennode, dtype=np.bool_) # 初期化用　ただし使いたくない。
        
        self.nodes_bool_start = np.zeros(lennode, dtype=np.bool_)
        self.nodes_bool_before_start = np.zeros(lennode, dtype=np.bool_)
        self.nodes_route = np.zeros(lennode, dtype=np.float32)
        self.nodes_num_robot_route = np.zeros(lennode, dtype=np.int16)

        self.nodes_position_x = np.zeros(lennode, dtype=np.int16)
        self.nodes_position_y = np.zeros(lennode, dtype=np.int16)
        

        self.nodes_bool_pick = np.zeros(lennode, dtype=np.bool_)
        self.nodes_bool_task = np.zeros(lennode, dtype=np.bool_)
        
        self.nodes_route_an_agent = np.zeros((NUM_AGENTS, lennode), dtype=np.float32)
        self.nodes_bool_route_an_agent = np.zeros((NUM_AGENTS, lennode), dtype=np.bool_)
        
        self.zeros_agents = np.zeros(NUM_AGENTS, dtype=np.int16)        # 初期化用　ただし使いたくない。
        self.agents_start_node = np.zeros(NUM_AGENTS, dtype=np.int16)   
        self.agents_before_start_node = np.zeros(NUM_AGENTS, dtype=np.int16)   
        self.agents_task_node = np.zeros(NUM_AGENTS, dtype=np.int16)
        
        self.knn_agents = np.zeros((NUM_AGENTS, KNN_AGENTS), dtype=np.int16)
        self.knn_agents_distanse = np.zeros((NUM_AGENTS, KNN_AGENTS), dtype=np.float32)

        for i in range(len(self.agents_before_start_node)):
            self.agents_before_start_node[i] = -1
        
        self.register_position()
        
    def register_position(self):
        for node in range(self.len_node):
            x, y = self.position_dict[node]
            self.nodes_position_x[node]=x
            self.nodes_position_y[node]=y


    def register_node(self, robot_id, node, agents_node, nodes_bool):
        nodes_bool[agents_node[robot_id]] = False
        agents_node[robot_id] = node
        nodes_bool[agents_node[robot_id]] = True
    
    def register_route_an_agent(self, robot_id, nodes_route):
        #self.nodes_route -= self.nodes_route_an_agent[robot_id][:]
        self.nodes_route_an_agent[robot_id][:] = nodes_route[:]
        self.nodes_bool_route_an_agent[robot_id][:] = nodes_route > 0
        #self.nodes_route += self.nodes_route_an_agent[robot_id][:]
    
    def register_bool_pick(self, robot_id, pick_timer, state_node):
        if pick_timer > 0:
            self.nodes_bool_pick[state_node] = True
        else:
            self.nodes_bool_pick[state_node] = False
    
    def register_robot_data(self, robot_id, robot_env):
        #print("graphDBのregister_robot_dataが呼び出されました。")
        #"""
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                state_node, task_id, pick_timer, task_timer, priority = robot_env.curr_state
                dest_node = robot_env.tasks_node[task_id]
            else:
                state_node, task_node, pick_timer, task_timer, priority = robot_env.curr_state
                dest_node = task_node
        else:
            state_node, task_id, pick_timer, task_timer, priority = robot_env.curr_state
            dest_node = robot_env.tasks_node[task_id]

        if state_node == dest_node and pick_timer==0:
            if TASK_ASSIGN_MODE:
                if not SINGLE_TASK_MODE:
                    dest_node = robot_env.assigned_task.assigned_task_checker[robot_id]
            else:
                task_id = task_id + 1
                dest_node = robot_env.tasks_node[task_id]
        #"""
        #state_node, task_id, pick_timer, task_timer, priority = robot_env.curr_state
        #dest_node = robot_env.tasks_node[task_id]
        #self.register_start_node(robot_id, state_node)
        #self.register_task_node(robot_id, dest_node)
        if self.agents_before_start_node[robot_id] == -1:
            before_state_node = state_node
        else:
            before_state_node = self.agents_start_node[robot_id]
            
        #self.register_node(robot_id, before_state_node, self.agents_before_start_node, self.nodes_bool_before_start) # start_node # NODE_INFO_BEFORE_STATE
        
        self.register_node(robot_id, state_node, self.agents_start_node, self.nodes_bool_start) # start_node
        self.register_node(robot_id, dest_node, self.agents_task_node, self.nodes_bool_task) # task_node
        self.register_route_an_agent(robot_id, robot_env.cnt_data[0:self.len_node])
        self.register_bool_pick(robot_id, pick_timer, state_node)
    
    def make_nodes_route(self):
        #self.nodes_route[:] = self.zeros_node[:]
        self.nodes_route[:] = np.sum(self.nodes_route_an_agent, axis=0)[:]
    
    def make_nodes_num_robot_route(self):
        self.nodes_num_robot_route[:] = np.sum(self.nodes_bool_route_an_agent, axis=0)[:]
    
    def make_nodes_common_data(self):
        self.make_nodes_route()
        self.make_nodes_num_robot_route()

    def make_knn_data(self, env_num_agents):
        for robot_id in range(env_num_agents):
            self.knn_agents[robot_id][:] = self.get_knn_agents_array(robot_id, env_num_agents)[:]
            self.knn_agents_distanse[robot_id][:] = self.get_knn_agents_distanse_array(robot_id)[:]#これはノード情報に入れられない。
            

    def get_node_common_data(self, node_id):
        node_data_info = np.zeros(NODE_DATA_COMMON_DIM, dtype=np.float32)
        node_data_info[NODE_INFO_COMMON_START] = self.nodes_bool_start[node_id]
        node_data_info[NODE_INFO_COMMON_ROUTE] = self.nodes_route[node_id]
        node_data_info[NODE_INFO_COMMON_NUMROBOT] = self.nodes_num_robot_route[node_id]

        return node_data_info
    
    #@profile
    def get_node_common_part_data(self, robot_id, node_id):
        #robots = self.get_knn_agents_array(robot_id)
        robots = self.knn_agents[robot_id]
        node_data_info = np.zeros(NODE_DATA_COMMON_DIM, dtype=np.float32)
        
        if NODE_INFO_COMMON_START is not None:
            node_data_info[NODE_INFO_COMMON_START] = self.nodes_bool_start[node_id]
        
        if NODE_INFO_COMMON_ROUTE is not None:
            node_data_info[NODE_INFO_COMMON_ROUTE] = np.sum(self.nodes_route_an_agent.T[node_id][robots]) / KNN_AGENTS
            #for i in range(KNN_AGENTS):
            #    if self.knn_agents_distanse[robot_id][i] != 0:
            #        node_data_info[NODE_INFO_COMMON_ROUTE] = self.nodes_route_an_agent.T[node_id][robots[i]]
            #node_data_info[NODE_INFO_COMMON_ROUTE] = node_data_info[NODE_INFO_COMMON_ROUTE] / KNN_AGENTS
            
        if NODE_INFO_COMMON_NUMROBOT is not None:
            node_data_info[NODE_INFO_COMMON_NUMROBOT] = np.sum(self.nodes_bool_route_an_agent.T[node_id][robots]) / KNN_AGENTS
            #for i in range(KNN_AGENTS):
            #    if self.knn_agents_distanse[robot_id][i] != 0:
            #        node_data_info[NODE_INFO_COMMON_NUMROBOT] = self.nodes_bool_route_an_agent.T[node_id][robots[i]]
            #node_data_info[NODE_INFO_COMMON_NUMROBOT] = node_data_info[NODE_INFO_COMMON_NUMROBOT] / KNN_AGENTS
        
        if NODE_INFO_COMMON_PICKER_BOOL is not None:
            node_data_info[NODE_INFO_COMMON_PICKER_BOOL] = self.nodes_bool_pick[node_id]
        
        #print(f"route{np.sum(self.nodes_route_an_agent.T[node_id][robots])}")
        #print(f"num_robot{np.sum(self.nodes_bool_route_an_agent.T[node_id][robots])}")
        #node_data_info[NODE_INFO_COMMON_NUMROBOT] = np.sum(self.nodes_bool_route_an_agent.T[node_id][robots])
        if POSMODE:
            node_data_info[NODE_INFO_COMMON_POSX] = self.nodes_position_x[node_id]
            node_data_info[NODE_INFO_COMMON_POSY] = self.nodes_position_y[node_id]
        return node_data_info
    
    #@profile
    def get_node_each_data(self, robot_id, node_id):
        node_data_info = np.zeros(NODE_DATA_EACH_DIM, dtype=np.float32)
        if NODE_INFO_STATE is not None:
            node_data_info[NODE_INFO_STATE] = 1.0 if self.agents_start_node[robot_id] == node_id else 0.0
        if NODE_INFO_ROUTE is not None:
            node_data_info[NODE_INFO_ROUTE] = self.nodes_route_an_agent[robot_id][node_id]
        
        #node_data_info[NODE_INFO_BEFORE_STATE] = 1.0 if self.agents_before_start_node[robot_id] == node_id else 0.0
        
        return node_data_info
    
    #@profile
    def get_node_data(self, robot_id, node_id):
        node_data_info = np.zeros(NODE_DATA_DIM, dtype=np.float32)
        node_data_info[:NODE_DATA_EACH_DIM] = self.get_node_each_data(robot_id, node_id)[:]
        if PARTIAL_AGENT_MODE or LOCALGCNMODE:
            node_data_info[NODE_DATA_EACH_DIM:] = self.get_node_common_part_data(robot_id,node_id)
        else:
            node_data_info[NODE_DATA_EACH_DIM:] = self.get_node_common_data(node_id)
        return node_data_info

    def get_knn_agents_array(self, robot_id, env_num_agents):
        return np.argsort(self.node_distances[self.agents_start_node[robot_id], self.agents_start_node][:env_num_agents])[:KNN_AGENTS]    # こっちが性能高め
        #return np.argsort(NODE_PHYSICAL_DISTANCES[self.agents_start_node[robot_id], self.agents_start_node])[:KNN_AGENTS] # 性能が低い
    
    def get_knn_reward_agents_array(self, robot_id, env_num_agents):
        return np.argsort(self.node_distances[self.agents_start_node[robot_id], self.agents_start_node][:env_num_agents])[:KNN_REWARD_AGENTS]    # こっちが性能高め
        #return np.argsort(NODE_PHYSICAL_DISTANCES[self.agents_start_node[robot_id], self.agents_start_node])[:KNN_AGENTS] # 性能が低い

    def get_knn_agents_distanse_array(self, robot_id):
        knn_start_node = np.array([self.agents_start_node[ka] for ka in self.knn_agents[robot_id]])
        dist_knn = self.node_distances[knn_start_node[0], knn_start_node]
        #print(f"ID:{robot_id}")
        #print(f"SN:{self.agents_start_node[robot_id]}")
        #print(f"KNN:{self.knn_agents[robot_id]}")
        #print(f"KNN_S:{knn_start_node}")
        #print(f"DKNN:{dist_knn}")
        #print(f"ret:{1/(1+dist_knn)}")
        dist_knn[dist_knn>DISTANCE] = DISTANCE
        return (DISTANCE-dist_knn)/DISTANCE
        return 2/(1+dist_knn)
    
    ###### for GNN ###########
    def get_gnn_node_data(self, robot_id, nodes):
        node_info = np.zeros(len(nodes)*NODE_DATA_DIM, dtype=np.float32)
        
        for i, node in enumerate(nodes):
            node_info[i*NODE_DATA_DIM:(i+1)*NODE_DATA_DIM] = self.get_node_data(robot_id, node)[:]
            #print(node)
            #print(node_info[i*NODE_DATA_DIM:(i+1)*NODE_DATA_DIM])
        return node_info
    
    def get_node_id_from(self, state_node, num):
        return self.bfs_get_node(state_node, num, self.connect_dict)
    
    def get_node_id_to(self, dest_node, num):
        return self.bfs_get_node(dest_node, num, self.connect_to_dict)
    
    def bfs_get_node(self, start_node, num, connect):
        nodes_id_list = []
        node_queue = deque([start_node])
        visited = set()
        
        while node_queue and len(nodes_id_list) < num:
            current_node = node_queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                nodes_id_list.append(current_node)
                node_queue.extend(connect[current_node])

        return np.array(nodes_id_list)
