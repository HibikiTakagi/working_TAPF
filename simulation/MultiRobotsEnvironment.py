import numpy as np
import random
import cv2
import gymnasium as gym
import time

#from numba import jit, prange
#from numba.experimental import jitclass
#from multiprocessing import Pool

from gymnasium import spaces
from graphDB import Graph, NODE_DATA_DIM, NODE_DATA_EACH_DIM

from RobotEnvironment import RobotEnvironment, draw_text
from CONST import (
    PENALTY_PRIVENT_COLLISION, NUM_AGENTS, MODE_AGENT, MAX_TIMESTEP, 
    PENALTY_NO_OP, PENALTY_PRIORITY,
    FULLNODEDATAMODE, PARTIALMODE, SINGLE_AGENT_MODE,PARTIAL_AGENT_MODE, LOCALGCNMODE, 
    #NUM_NODES_OF_LOCAL_GRAPH_TO, NUM_NODES_OF_LOCAL_GRAPH_FROM, 
    NUM_NODES_OF_LOCAL_GRAPH_TO_START, NUM_NODES_OF_LOCAL_GRAPH_FROM_START, NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST, NUM_NODES_OF_LOCAL_GRAPH_TO_DEST,
    NUM_NODES_OF_LOCAL_GRAPH_START, NUM_NODES_OF_LOCAL_GRAPH_DEST,
    KNN_MODE, KNN_AGENTS, DIST_MODE, KNN_REWARD_AGENTS
    )
#from map_maker import IS_RANDOM, LEN_NODE,  ACTION_LIST, ROUTE, TEST_MODE
from map_maker import IMAGE_WIDTH, TASK_ASSIGN_MODE, SINGLE_TASK_MODE

NUM_PROCESS=16
IS_MULTI_PROCESS = False


DIM_ELSE_SINGLE_STATE = 2
#STATE_ID_NEXT_PICKING = 3
STATE_ID_PRIORITY = 2
STATE_ID_PICK_TIMER = 1


def unwrap_self_f(arg):
    # メソッドfをクラスメソッドとして呼び出す関数
    return MultiRobotsEnvironment.single_phase3_transition(*arg)

class MultiRobotsEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode_id):
        super(MultiRobotsEnvironment, self).__init__()
        self.mode_id = mode_id
    
    def map_init_reset(self, num_agents, len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, actionlist, picker_node, is_pickup, assigned_task):
        self.len_node = len_node
        self.node_distances = node_distances
        self.connect_dict = connect_dict
        self.connect_to_dict = connect_to_dict
        self.coord_list = coord_list
        self.weights = weights
        self.actionlist = actionlist
        self.pickernode = picker_node
        
        self.each_info_dim = 1
        self.info_dim = 1
        
        self.graph = Graph(len_node=self.len_node, node_distances=self.node_distances, connectdict=self.connect_dict, connect_to_dict=self.connect_to_dict, coord_list=self.coord_list)

        #self.num_agents = MODE_AGENT[self.mode_id]
        self.num_agents = num_agents
        self.robot_envs = [RobotEnvironment(i, self.len_node, self.connect_dict, self.coord_list, self.node_distances, self.weights, self.actionlist, self.pickernode, is_pickup, assigned_task) for i in range(self.num_agents)]

        self.action_dim = (self.num_agents,)
        self.action_space = gym.spaces.Box(
            low=0, 
            high=self.robot_envs[0].action_dim-1, 
            shape=self.action_dim, 
            dtype=np.int8
            )
        self.full_obs_dim = self.robot_envs[0].obs_dim * self.num_agents # fullobsdim
        if FULLNODEDATAMODE:
            self.obs_dim = self.full_obs_dim # modeによって異なる
        elif PARTIALMODE:
            self.nodedata_dim = NODE_DATA_DIM
            self.single_obs_dim = (self.robot_envs[0].action_dim + 1) * self.nodedata_dim + DIM_ELSE_SINGLE_STATE
            # start, route, allstart, allroute, allnumrobot
            self.obs_dim = self.single_obs_dim * self.num_agents
        elif PARTIAL_AGENT_MODE:
            self.nodedata_dim = NODE_DATA_DIM
            self.num_node = (self.robot_envs[0].action_dim + 1)
            
            self.single_obs_dim = (self.robot_envs[0].action_dim + 1) * self.nodedata_dim + DIM_ELSE_SINGLE_STATE
            if DIST_MODE:
                self.single_obs_dim = (self.robot_envs[0].action_dim + 1) * self.nodedata_dim + DIM_ELSE_SINGLE_STATE + KNN_AGENTS
            # start, route, partstart, partroute, partnumrobot
            #self.obs_dim = self.single_obs_dim * self.num_agents
            self.each_obs_dim = self.single_obs_dim * self.num_agents
            self.obs_dim = (self.single_obs_dim * KNN_AGENTS)* self.num_agents
            if DIST_MODE:
                self.obs_dim = (self.single_obs_dim * KNN_AGENTS + KNN_AGENTS)* self.num_agents
                
        elif LOCALGCNMODE:
            #self.num_nodes_of_local_graph_from = NUM_NODES_OF_LOCAL_GRAPH_FROM
            #self.num_nodes_of_local_graph_to = NUM_NODES_OF_LOCAL_GRAPH_TO

            
            """
            self.single_obs_dim = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_FROM \
                                + NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_TO \
                                + NUM_NODES_OF_LOCAL_GRAPH_FROM \
                                + NUM_NODES_OF_LOCAL_GRAPH_TO \
                                + 2
            """
            self.num_node = (NUM_NODES_OF_LOCAL_GRAPH_FROM_START + NUM_NODES_OF_LOCAL_GRAPH_TO_START) +  (NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST + NUM_NODES_OF_LOCAL_GRAPH_TO_DEST)
            self.nodedata_dim = NODE_DATA_DIM
            
            self.single_obs_dim = NODE_DATA_DIM * (NUM_NODES_OF_LOCAL_GRAPH_FROM_START + NUM_NODES_OF_LOCAL_GRAPH_TO_START) \
                                + NODE_DATA_DIM * (NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST + NUM_NODES_OF_LOCAL_GRAPH_TO_DEST) \
                                + DIM_ELSE_SINGLE_STATE\
                                #+ NUM_NODES_OF_LOCAL_GRAPH_FROM_START + NUM_NODES_OF_LOCAL_GRAPH_TO_START\
                                #+ NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST + NUM_NODES_OF_LOCAL_GRAPH_TO_DEST\
            
            self.single_info_dim = + NUM_NODES_OF_LOCAL_GRAPH_FROM_START + NUM_NODES_OF_LOCAL_GRAPH_TO_START\
                                    + NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST + NUM_NODES_OF_LOCAL_GRAPH_TO_DEST
            self.each_info_dim = self.single_info_dim * self.num_agents
            self.info_dim = (self.single_info_dim * KNN_AGENTS) * self.num_agents
            
            # start, route, partstart, partroute, partnumrobot
            #self.obs_dim = self.single_obs_dim * self.num_agents
            self.each_obs_dim = self.single_obs_dim * self.num_agents
            self.obs_dim = (self.single_obs_dim * KNN_AGENTS)* self.num_agents
            if DIST_MODE:
                self.obs_dim = (self.single_obs_dim * KNN_AGENTS + KNN_AGENTS)* self.num_agents

        self.observation_dim = (self.obs_dim,)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=self.observation_dim, 
            dtype=np.float32
            )
        self.reward_range = (-float('inf'), float('inf'))

        self.num_steps = None
        self.cnt_noop = None
        self.cnt_privent_collision = None
        self.cnt_all_privent_collision = None
        self.rewards = None
        self.reward = None

        self.fig = None

        if SINGLE_AGENT_MODE:
            self.net_act_dim = self.robot_envs[0].action_dim
            if KNN_MODE:
                self.net_state_dim = self.single_obs_dim * KNN_AGENTS
                if DIST_MODE:
                    self.net_state_dim = self.single_obs_dim * KNN_AGENTS + KNN_AGENTS
            else:
                self.net_state_dim = self.obs_dim
        else:
            self.net_act_dim = self.robot_envs[0].action_dim * self.num_agents
            self.net_state_dim = self.obs_dim
        #"""
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        
        self.zeros_each_observe = np.zeros(self.each_obs_dim, dtype=np.float32)
        self.each_observation = np.zeros(self.each_obs_dim, dtype=np.float32)
        
        self.zeros_each_info = np.zeros(self.each_info_dim, dtype=np.int16)
        self.each_info = np.zeros(self.each_info_dim, dtype=np.int16)
        self.node_info = np.zeros(self.info_dim, dtype=np.int16)
        
        self.zeros_observe = np.zeros(self.obs_dim, dtype=np.float32)
        self.observation = np.zeros(self.obs_dim, dtype=np.float32)
        
        self.zeros_full_observe = np.zeros(self.full_obs_dim, dtype=np.float32)
        self.full_observation = np.zeros(self.full_obs_dim, dtype=np.float32)

        self.zeros_node = np.zeros(self.len_node, dtype=np.float32)
        #self.nodes_start = np.zeros(self.len_node, dtype=np.float32)
        #self.nodes_route = np.zeros(self.len_node, dtype=np.float32)
        #self.nodes_num_robot = np.zeros(self.len_node, dtype=np.float32)

        self.reserve_node = np.zeros(self.len_node, dtype=np.float32)
        self.zero_state_nodes = np.zeros(NUM_AGENTS, dtype=np.int32)
        self.post_state_nodes = np.zeros(NUM_AGENTS, dtype=np.int32)
        self.state_nodes = np.zeros(NUM_AGENTS, dtype=np.int32)
        self.zero_priority = np.zeros(NUM_AGENTS, dtype=np.float32)
        self.priority = np.zeros(NUM_AGENTS, dtype=np.float32)

        self.zeros_replay_obs = np.zeros(self.net_state_dim, dtype=np.float32)
        self.replay_obs = np.zeros(self.net_state_dim, dtype=np.float32)
        self.replay_obs_next = np.zeros(self.net_state_dim, dtype=np.float32)

        self.zeros_knn_obs = np.zeros(KNN_AGENTS, dtype=np.float32)
        self.knn_obs = np.zeros(KNN_AGENTS, dtype=np.float32)
        self.knn_obs_next = np.zeros(KNN_AGENTS, dtype=np.float32)


        #self.zeros_gnn_nodes_from = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_FROM, dtype=np.int32)
        #self.gnn_nodes_from = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_FROM, dtype=np.int32)
        #self.zeros_gnn_nodes_to = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_TO, dtype=np.int32)
        #self.gnn_nodes_to = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_TO, dtype=np.int32)

        self.zeros_gnn_nodes_from_start = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_FROM_START, dtype=np.int32)
        self.gnn_nodes_from_start = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_FROM_START, dtype=np.int32)
        self.zeros_gnn_nodes_to_start = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_TO_START, dtype=np.int32)
        self.gnn_nodes_to_start = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_TO_START, dtype=np.int32)
        self.zeros_gnn_nodes_from_dest = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST, dtype=np.int32)
        self.gnn_nodes_from_dest = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST, dtype=np.int32)
        self.zeros_gnn_nodes_to_dest = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_TO_DEST, dtype=np.int32)
        self.gnn_nodes_to_dest = np.zeros(NUM_NODES_OF_LOCAL_GRAPH_TO_DEST, dtype=np.int32)

        #if IS_MULTI_PROCESS:
        #    self.pool = Pool(processes=NUM_PROCESS)
        
    def reset(self):
        #print("MultiRobotsEnvironmentのresetが呼び出されました。")
        start_node_idxes = random.sample(list(range(self.len_node)), self.num_agents)
        for i in range(self.num_agents):
            self.robot_envs[i].multi_reset(state_node=start_node_idxes[i])
        self.num_steps = 0
        self.cnt_noop = 0
        self.cnt_privent_collision = 0
        self.cnt_all_privent_collision = 0
        self.rewards = np.array([0.0 for _ in range(self.num_agents)])
        self.processed_rewards = np.array([0.0 for _ in range(self.num_agents)])
        self.reward = 0.0
        
        for i in range(self.num_agents):
            self.graph.register_robot_data(i, self.robot_envs[i])
        self.graph.make_nodes_common_data()

        self.make_observe()
        self.state[:] = self.observation[:]

        for i in range(self.num_agents):
            state_node, _, _, _, _ = self.robot_envs[i].curr_state
            self.state_nodes[i] = state_node
        
        info = {"nodeinfo":self.node_info.copy()}
        #print(f"MultiRobotsEnvironmentのresetの結果、stateは{self.state}となりました。")
        return self.state.copy(), info
    
    def step(self, action):
        observation, reward = self.transition(action) # 99% of step is this process
        terminated = self.is_terminal
        truncated = (MAX_TIMESTEP<=self.num_steps) and (not terminated)
        info = {"nodeinfo":self.node_info.copy()}
        self.num_steps += 1
        return observation.copy(), reward, terminated, truncated, info
    
    def virtual_step(self, action):
        observation, reward = self.virtual_transition(action) # 99% of step is this process
        terminated = self.is_terminal
        truncated = (MAX_TIMESTEP<=self.num_steps) and (not terminated)
        info = {} #{"nodeinfo":self.node_info.copy()} # future work
        self.num_steps += 1
        return observation.copy(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        ####AGV####
        self.fig = self.robot_envs[0].draw_graph()
        ####UAV####
        #self.fig = self.robot_envs[0].draw_graph_uav()
        ###########
        
        for i in range(self.num_agents):
            ####AGV####
            self.robot_envs[i].draw_robot(self.fig)
            ####UAV####
            #self.robot_envs[i].draw_robot_uav(self.fig)
            ###########
        #    self.robot_envs[i].draw_data(self.fig)
        #self.robot_envs[0].draw_robot(self.fig)
        self.robot_envs[0].draw_data(self.fig)
        
        self.draw_data(self.fig)
        fig = cv2.cvtColor(self.fig, cv2.COLOR_BGR2RGB)
        return fig
        pass
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        pass

    # 以下追加
    @property
    def is_terminal(self):
        return False
        """
        ret = True
        for i in range(self.num_agents):
            ret = ret and self.robot_envs[i].is_terminal
        return ret
        """
    #"""
    #@profile
    #@jit(nopython=False, parallel=True)
    #@profile
    def transition(self, action):
        #print(f"MultiRobotsEnvironmentのtransitionが呼び出されました。")
        num_noop_in_step = 0
        num_privent_collision_in_step = 0
        sum_violate_priority = 0.0
        
        self.reserve_node[:] = self.zeros_node[:]
        self.post_state_nodes[:] = self.zero_state_nodes[:]
        self.state_nodes[:] = self.zero_state_nodes[:]

        self.priority[:] = self.zero_priority[:]
        
        ### Phase 1 : Reserve
        for i in range(self.num_agents):
            robot = self.robot_envs[i]
            pre_act = action[i]
            # state_node, task_id, pick_timer, task_timer, priority = robot.curr_state
            state_node, _, pick_timer, task_timer, priority = robot.curr_state
            self.priority[i] = priority
            self.reserve_node[state_node] += 1
            #post_state_node, self.priority[i] = robot.pretransition_state(pre_act)
            post_state_node = robot.pretransition_state(pre_act)
            self.post_state_nodes[i] = post_state_node
            self.state_nodes[i] = state_node
            if state_node != post_state_node:
                self.reserve_node[post_state_node] += 1
        
        ### Phase2 : Abort
        collision_nodes,  = np.where(self.reserve_node > 1)
        num_privent_collision_in_step = int(self.post_state_nodes[0] in collision_nodes and self.post_state_nodes[0] != self.state_nodes[0])
        num_all_privent_collision_in_step = self.collision_count(collision_nodes=collision_nodes)
        for col_node in collision_nodes:
            col_agent_idx, = np.where(self.post_state_nodes[:self.num_agents]==col_node) # if #agent is random, post_state_nodes[numagents:] is 0. so fantom collision will happen.
            col_priority = self.priority[col_agent_idx]
            if col_node in self.state_nodes:
                #print("=====")
                #print(self.reserve_node[0])
                #print(collision_nodes)
                #print(self.post_state_nodes)
                #print(col_node)
                #print(self.num_agents)
                #print(col_agent_idx)
                action[col_agent_idx] = 0
                continue
            agent_max_priority = col_agent_idx[col_priority.argmax()]
            for i in col_agent_idx:
                if i != agent_max_priority:
                    action[i] = 0

        ### Phase3 : Commit
        if IS_MULTI_PROCESS:
            self.multi_phase3_transition(action)
        else:
            self.phase3_transition(action) # GNN 35.9 
        
        self.cnt_privent_collision += num_privent_collision_in_step
        self.cnt_all_privent_collision += num_all_privent_collision_in_step
        self.cnt_noop += num_noop_in_step

        reward = self._reward_function(
            num_privent_collision_in_step,
            num_noop_in_step,
            sum_violate_priority
            )
        self.reward = reward
    
        self.make_observe() # GNN 54.7
        self.state[:] = self.observation[:]
        
        return self.state, reward
    #"""

    #@profile # if GNN ignore time
    def phase3_transition(self, action):
        for i in range(self.num_agents):
            robot = self.robot_envs[i]
            _, rw = robot.transition(action=action[i]) # 99.2% but per hit is very small (Hit is many)
            self.rewards[i] = rw
    
    def phase3_virtual_transition(self, action):
        for i in range(self.num_agents):
            robot = self.robot_envs[i]
            _, rw = robot.virtual_transition(action=action[i]) # 99.2% but per hit is very small (Hit is many)
            self.rewards[i] = rw
    
    ## した２つは上をmultiprocessで並列化するためのコード
    @staticmethod
    def single_phase3_transition(robot, action):
        _, rw = robot.transition(action=action)
        return rw, robot
    
    #@profile
    def multi_phase3_transition(self, action):
        values = [(self.robot_envs[i], action[i]) for i in range(self.num_agents)] 
        result = self.pool.map(unwrap_self_f, values, 1+self.num_agents//(NUM_PROCESS))[:]
        #result = pool.map(self.single_phase3_transition, values)[:]
        self.robot_envs = [r[1] for r in result]
        self.rewards = np.array([r[0] for r in result])
        #print(self.num_steps, self.rewards)

    def _reward_function(self, col, noop, pri):
        processed_reward = (
            self.rewards[0] / KNN_REWARD_AGENTS
            #+ np.sum(self.rewards[self.graph.get_knn_agents_array(0, self.num_agents)])/KNN_AGENTS
            + np.sum(self.rewards[self.graph.get_knn_reward_agents_array(0, self.num_agents)])/KNN_REWARD_AGENTS
            #+ np.dot(self.rewards[self.graph.get_knn_agents_array(0, self.num_agents)], self.graph.knn_agents_distanse[0])/KNN_AGENTS
            - col * PENALTY_PRIVENT_COLLISION
            - noop * PENALTY_NO_OP
            - pri/self.num_agents * PENALTY_PRIORITY
        ) 
        #print(f"{self.rewards[0]},{np.sum(self.rewards[self.graph.get_knn_agents_array(0, self.num_agents)])/KNN_AGENTS}, {col * PENALTY_PRIVENT_COLLISION}")
        return processed_reward

    def _reward_function_id(self, idx, col, noop, pri):
        processed_reward = (
            self.rewards[idx] / KNN_REWARD_AGENTS
            #+ np.sum(self.rewards[self.graph.get_knn_agents_array(0, self.num_agents)])/KNN_AGENTS
            + np.sum(self.rewards[self.graph.get_knn_reward_agents_array(idx, self.num_agents)])/KNN_REWARD_AGENTS
            #+ np.dot(self.rewards[self.graph.get_knn_agents_array(0, self.num_agents)], self.graph.knn_agents_distanse[0])/KNN_AGENTS
            - col * PENALTY_PRIVENT_COLLISION
            - noop * PENALTY_NO_OP
            - pri/self.num_agents * PENALTY_PRIORITY
        ) 
        #print(f"{self.rewards[0]},{np.sum(self.rewards[self.graph.get_knn_agents_array(0, self.num_agents)])/KNN_AGENTS}, {col * PENALTY_PRIVENT_COLLISION}")
        return processed_reward    

    def collision_count(self, collision_nodes):
        num_all_privent_collision_in_step = 0
        for i in range(self.num_agents):
            if self.post_state_nodes[i] in collision_nodes and self.post_state_nodes[i] != self.state_nodes[i]:
                num_all_privent_collision_in_step += 1
        return num_all_privent_collision_in_step
    
    def virtual_transition(self, action):
        num_noop_in_step = 0
        num_privent_collision_in_step = 0
        sum_violate_priority = 0.0
        
        self.reserve_node[:] = self.zeros_node[:]
        self.post_state_nodes[:] = self.zero_state_nodes[:]
        self.state_nodes[:] = self.zero_state_nodes[:]

        self.priority[:] = self.zero_priority[:]
        
        ### Phase 1 : Reserve
        for i in range(self.num_agents):
            robot = self.robot_envs[i]
            pre_act = action[i]
            state_node, task_id, pick_timer, task_timer, priority = robot.curr_state
            self.priority[i] = priority
            self.reserve_node[state_node] += 1
            #post_state_node, self.priority[i] = robot.pretransition_state(pre_act)
            post_state_node = robot.pretransition_state(pre_act)
            self.post_state_nodes[i] = post_state_node
            self.state_nodes[i] = state_node
            if state_node != post_state_node:
                self.reserve_node[post_state_node] += 1
        
        ### Phase2 : Abort
        collision_nodes,  = np.where(self.reserve_node > 1)
        num_privent_collision_in_step = int(self.post_state_nodes[0] in collision_nodes and self.post_state_nodes[0] != self.state_nodes[0])
        num_all_privent_collision_in_step = self.collision_count(collision_nodes=collision_nodes)
        for col_node in collision_nodes:
            col_agent_idx, = np.where(self.post_state_nodes[:self.num_agents]==col_node) # if #agent is random, post_state_nodes[numagents:] is 0. so fantom collision will happen.
            col_priority = self.priority[col_agent_idx]
            if col_node in self.state_nodes:
                action[col_agent_idx] = 0
                continue
            agent_max_priority = col_agent_idx[col_priority.argmax()]
            for i in col_agent_idx:
                if i != agent_max_priority:
                    action[i] = 0
        
        ### Phase3 : Commit
        self.phase3_virtual_transition(action) # GNN 35.9 
        
        self.cnt_privent_collision += num_privent_collision_in_step
        self.cnt_all_privent_collision += num_all_privent_collision_in_step
        self.cnt_noop += num_noop_in_step

        reward = self._reward_function(
            num_privent_collision_in_step,
            num_noop_in_step,
            sum_violate_priority
            )
        self.reward = reward
        
        self.make_observe() # GNN 54.7
        self.state[:] = self.observation[:]
        
        return self.state, reward

    #@profile
    def make_observe(self):
        for i in range(self.num_agents):
            self.graph.register_robot_data(i, self.robot_envs[i])

        if FULLNODEDATAMODE:
            self.graph.make_nodes_common_data()
            self.make_observe_fullnode()
        elif PARTIALMODE:
            self.graph.make_nodes_common_data()
            self.make_observe_partnode()
        elif PARTIAL_AGENT_MODE:
            self.make_observe_part_agent_node()
        elif LOCALGCNMODE:
            self.make_observe_gcn_part_agent_node()

        #print(self.observation)
        if KNN_MODE:
            self.knn_state_next_maker(self.each_observation)
            self.graph.make_knn_data(self.num_agents)
            self.convert_and_add_observe(self.each_observation, self.each_info)
            self.knn_state_maker(self.each_observation)
            #self.convert_and_add_observe(self.each_observation)
            #if PARTIAL_AGENT_MODE:
            #    self.make_add_knninfo_observe_part_agent_node()
    
    #@profile
    def convert_and_add_observe(self, state, info):
        state = state.reshape(self.num_agents, -1)
        info = info.reshape(self.num_agents, -1)
        robot_offset = (self.single_obs_dim*KNN_AGENTS)
        robot_info_offset = (self.single_info_dim*KNN_AGENTS)
        if DIST_MODE:
            robot_offset = (self.single_obs_dim*KNN_AGENTS) + KNN_AGENTS
        for robot_idx in range(self.num_agents):    
            self.observation[robot_idx*robot_offset:robot_idx*robot_offset+self.single_obs_dim*KNN_AGENTS] = self.single_state_maker(state, robot_idx)[:]
            self.node_info[robot_idx*robot_info_offset:robot_idx*robot_info_offset+self.single_info_dim*KNN_AGENTS] = self.single_info_maker(info, robot_idx)[:]
            if DIST_MODE:
                self.observation[robot_idx*robot_offset+self.single_obs_dim*KNN_AGENTS:(robot_idx+1)*robot_offset] = self.graph.knn_agents_distanse[robot_idx][:]
        
    #@profile
    def single_state_maker(self, state, idx):
        if KNN_MODE:
            data = state[self.graph.knn_agents[idx]]
            """
            for i in range(len(data)):
                #if self.graph.knn_agents_distanse[idx][i] == 0:
                #    # これはDQNなら早いが、GNNになった途端にノードが0に置き換わるバグを生み出すのでアウト
                #    data[i][:] = data[i][:]*self.graph.knn_agents_distanse[idx][i]
                #else:
                #    for j in range(self.num_node):
                #        node_offset = self.nodedata_dim*j
                #        #data[i][node_offset:node_offset+NODE_DATA_EACH_DIM] = data[i][node_offset:node_offset+NODE_DATA_EACH_DIM]*self.graph.knn_agents_distanse[idx][i]
                #        data[i][node_offset:node_offset+NODE_DATA_DIM] = data[i][node_offset:node_offset+NODE_DATA_DIM]*self.graph.knn_agents_distanse[idx][i]
                #    data[i][self.single_obs_dim-STATE_ID_PICK_TIMER] = data[i][self.single_obs_dim-STATE_ID_PICK_TIMER]*self.graph.knn_agents_distanse[idx][i]
                #    #data[i][:] = data[i][:]*self.graph.knn_agents_distanse[idx][i]
                
                for j in range(self.num_node):
                    node_offset = self.nodedata_dim*j
                    #data[i][node_offset:node_offset+NODE_DATA_EACH_DIM] = data[i][node_offset:node_offset+NODE_DATA_EACH_DIM]*self.graph.knn_agents_distanse[idx][i]
                    #data[i][node_offset:node_offset+NODE_DATA_DIM] = data[i][node_offset:node_offset+NODE_DATA_DIM]*self.graph.knn_agents_distanse[idx][i]
                    if self.graph.knn_agents_distanse[idx][i] == 0:
                        data[i][node_offset:node_offset+NODE_DATA_DIM] = data[i][node_offset:node_offset+NODE_DATA_DIM]*0
                #data[i][self.single_obs_dim-STATE_ID_PICK_TIMER] = data[i][self.single_obs_dim-STATE_ID_PICK_TIMER]*self.graph.knn_agents_distanse[idx][i]
                if self.graph.knn_agents_distanse[idx][i] == 0:
                    data[i][self.single_obs_dim-STATE_ID_PICK_TIMER] = data[i][self.single_obs_dim-STATE_ID_PICK_TIMER]*0
                
            #"""
            #print(data.shape)
            return np.concatenate(data)
        else:
            data = np.insert(np.delete(state, idx, axis=0),0, state[idx], axis=0)
            return np.concatenate(data)
    
    #@profile
    def single_info_maker(self, info, idx):
        if KNN_MODE:
            data = info[self.graph.knn_agents[idx]]
            """
            for i in range(len(data)):
                #if self.graph.knn_agents_distanse[idx][i] == 0:
                #    # これはDQNなら早いが、GNNになった途端にノードが0に置き換わるバグを生み出すのでアウト
                #    data[i][:] = data[i][:]*self.graph.knn_agents_distanse[idx][i]
                #else:
                #    for j in range(self.num_node):
                #        node_offset = self.nodedata_dim*j
                #        #data[i][node_offset:node_offset+NODE_DATA_EACH_DIM] = data[i][node_offset:node_offset+NODE_DATA_EACH_DIM]*self.graph.knn_agents_distanse[idx][i]
                #        data[i][node_offset:node_offset+NODE_DATA_DIM] = data[i][node_offset:node_offset+NODE_DATA_DIM]*self.graph.knn_agents_distanse[idx][i]
                #    data[i][self.single_obs_dim-STATE_ID_PICK_TIMER] = data[i][self.single_obs_dim-STATE_ID_PICK_TIMER]*self.graph.knn_agents_distanse[idx][i]
                #    #data[i][:] = data[i][:]*self.graph.knn_agents_distanse[idx][i]
                
                for j in range(self.num_node):
                    node_offset = self.nodedata_dim*j
                    #data[i][node_offset:node_offset+NODE_DATA_EACH_DIM] = data[i][node_offset:node_offset+NODE_DATA_EACH_DIM]*self.graph.knn_agents_distanse[idx][i]
                    #data[i][node_offset:node_offset+NODE_DATA_DIM] = data[i][node_offset:node_offset+NODE_DATA_DIM]*self.graph.knn_agents_distanse[idx][i]
                    if self.graph.knn_agents_distanse[idx][i] == 0:
                        data[i][node_offset:node_offset+NODE_DATA_DIM] = data[i][node_offset:node_offset+NODE_DATA_DIM]*0
                #data[i][self.single_obs_dim-STATE_ID_PICK_TIMER] = data[i][self.single_obs_dim-STATE_ID_PICK_TIMER]*self.graph.knn_agents_distanse[idx][i]
                if self.graph.knn_agents_distanse[idx][i] == 0:
                    data[i][self.single_obs_dim-STATE_ID_PICK_TIMER] = data[i][self.single_obs_dim-STATE_ID_PICK_TIMER]*0
                
            #"""
            #print(data.shape)
            return np.concatenate(data)
        else:
            data = np.insert(np.delete(info, idx, axis=0),0, info[idx], axis=0)
            return np.concatenate(data)
    
    #@profile
    def knn_state_maker(self, state):
        
        robot_offset = (self.single_obs_dim*KNN_AGENTS)
        if DIST_MODE:
            robot_offset = (self.single_obs_dim*KNN_AGENTS) + KNN_AGENTS
            
        state = state.reshape(self.num_agents, -1)
        self.replay_obs[:]=self.observation[:robot_offset]
        #self.replay_obs[:self.single_obs_dim * KNN_AGENTS] = np.concatenate(state[self.graph.knn_agents[0]])
        #if DIST_MODE:
        #    self.replay_obs[self.single_obs_dim * KNN_AGENTS:] = self.graph.knn_agents_distanse[0][:]

        #print(self.replay_obs==self.observation[:self.single_obs_dim * KNN_AGENTS])
        #print(f"self.replay_obs{self.replay_obs}")
        #print(f"self.observation[:self.single_obs_dim * KNN_AGENTS]{self.observation[:self.single_obs_dim * KNN_AGENTS]}")
    
    #@profile
    def knn_state_next_maker(self, state):
        state = state.reshape(self.num_agents, -1)
        #print(f"replay_obs:{self.replay_obs.shape}")
        #print(f"state:{state[self.graph.knn_agents[0]].shape}")
        self.replay_obs_next[:self.single_obs_dim * KNN_AGENTS] = self.single_state_maker(state, 0)
        
        if DIST_MODE:
            self.replay_obs_next[self.single_obs_dim * KNN_AGENTS:] = self.graph.knn_agents_distanse[0][:]
        
        #print(self.replay_obs_next==self.observation[:self.single_obs_dim * KNN_AGENTS]) # これは対象となる抜き出しAgentが変わるためできない。
        #robot_offset = (self.single_obs_dim*KNN_AGENTS)
        #if DIST_MODE:
        #    robot_offset = (self.single_obs_dim*KNN_AGENTS) + KNN_AGENTS
        #self.replay_obs_next[:robot_offset] = self.observation[:robot_offset]

    def make_full_observation(self):
        ## unavailable
        self.full_observation[:] = self.zeros_full_observe[:]
        for i in range(self.num_agents):
            state_node, task_id, pick_timer, task_timer, priority = self.robot_envs[i].next_state
            robot_offset = i * self.robot_envs[0].obs_dim
            self.full_observation[robot_offset + state_node] = 1
            #self.full_observation[robot_offset+self.len_node:robot_offset+self.len_node*2+2] = self.robot_envs[i].state[self.len_node:self.len_node*2+2].copy()
            self.full_observation[robot_offset+self.len_node:robot_offset+self.len_node*2] = self.robot_envs[i].state[0:self.len_node]
            self.full_observation[robot_offset+self.len_node:robot_offset+self.len_node*2 + STATE_ID_PRIORITY] = priority
            self.full_observation[robot_offset+self.len_node:robot_offset+self.len_node*2 + STATE_ID_PICK_TIMER] = pick_timer

    def make_observe_fullnode(self):
        ## unavailable
        self.make_full_observation()
        self.observation[:] = self.full_observation[:]

    #'''
    def make_observe_partnode(self):
        self.observation[:] = self.zeros_observe[:]
        
        for robot_id in range(self.num_agents):
            state_node, task_id, pick_timer, task_timer, priority = self.robot_envs[robot_id].next_state
            dest_node = self.robot_envs[robot_id].tasks_node[task_id]
            #if state_node == dest_node and pick_timer==0:
            #    task_id = task_id + 1
            #    dest_node = self.robot_envs[i].tasks_node[task_id]
            
            robot_offset = robot_id * self.single_obs_dim
            dest_id = len(self.actionlist[state_node])
            for j, node in enumerate(self.actionlist[state_node]):
                node_offset = robot_offset+self.nodedata_dim*j
                self.observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, node)[:]
            node_offset = robot_offset+self.nodedata_dim*dest_id 
            self.observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, dest_node)[:]

            self.observation[robot_offset+self.single_obs_dim-2] = priority
            self.observation[robot_offset+self.single_obs_dim-1] = pick_timer
            
        """
        print(f"{check_node_list}")
        print(self.observation[0:5])
        print(self.observation[5:10])
        print(self.observation[10:15])
        print(self.observation[15:20])
        print(f"{state_node}>{dest_node}")
        """
    #'''

    #'''
    #@profile
    def make_observe_part_agent_node(self):
        #KNN抜き出しをこっちで本音はやりたい。
        #しかし現状はDQLでやっている。
        self.observation[:] = self.zeros_observe[:]
        
        for robot_id in range(self.num_agents):
            state_node, task_id, pick_timer, task_timer, priority = self.robot_envs[robot_id].next_state
            dest_node = self.robot_envs[robot_id].tasks_node[task_id]
            
            robot_offset = robot_id * self.single_obs_dim
            dest_id = len(self.actionlist[state_node])
            for j, node in enumerate(self.actionlist[state_node]):
                node_offset = robot_offset+self.nodedata_dim*j
                self.each_observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, node)[:]
            node_offset = robot_offset+self.nodedata_dim*dest_id 
            self.each_observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, dest_node)[:]
            
            #self.each_observation[robot_offset+self.single_obs_dim-STATE_ID_NEXT_PICKING] = self.robot_envs[robot_id].tasks_is_picking[task_id]
            self.each_observation[robot_offset+self.single_obs_dim-STATE_ID_PRIORITY] = priority
            self.each_observation[robot_offset+self.single_obs_dim-STATE_ID_PICK_TIMER] = pick_timer
            
            #self.each_observation[robot_offset+self.single_obs_dim-2-KNN_AGENTS] = priority
            #self.each_observation[robot_offset+self.single_obs_dim-1-KNN_AGENTS] = pick_timer
    #'''
    def make_add_knninfo_observe_part_agent_node(self):
        for robot_id in range(self.num_agents):
            robot_offset = robot_id * self.single_obs_dim
            self.graph.get_knn_agents_distanse_array(robot_id)
            #self.each_observation[robot_offset+self.single_obs_dim-3-KNN_AGENTS:robot_offset+self.single_obs_dim-3] = self.graph.get_knn_agents_distanse_array(robot_id)[:]
    
    #@profile
    def make_observe_gcn_part_agent_node(self):
        #print("make_observe_gcn_part_agent_nodeが呼び出されました。")
        """
        start_nodedata, 
        dest_nodedata,

        node_id_FROM,
        node_id_TO,

        priority
        pick_timer
        """
        self.observation[:] = self.zeros_observe[:]
        
        for robot_id in range(self.num_agents):
            if TASK_ASSIGN_MODE:
                if SINGLE_TASK_MODE:
                    state_node, task_id, pick_timer, task_timer, priority = self.robot_envs[robot_id].next_state
                    dest_node = self.robot_envs[robot_id].tasks_node[task_id]
                else:
                    state_node, task_node, pick_timer, task_timer, priority = self.robot_envs[robot_id].next_state
                    dest_node = self.robot_envs[robot_id].task_node
            else:
                state_node, task_id, pick_timer, task_timer, priority = self.robot_envs[robot_id].next_state
                dest_node = self.robot_envs[robot_id].tasks_node[task_id]
            robot_offset = robot_id * self.single_obs_dim
            robot_info_offset = robot_id * self.single_info_dim
            """
            self.gnn_nodes_from[:] = self.graph.get_node_id_from(state_node, NUM_NODES_OF_LOCAL_GRAPH_FROM)[:]
            self.gnn_nodes_to[:] = self.graph.get_node_id_to(dest_node, NUM_NODES_OF_LOCAL_GRAPH_TO)[:]

            self.observation[robot_offset:\
                             robot_offset+NODE_DATA_DIM*NUM_NODES_OF_LOCAL_GRAPH_FROM] \
                                = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_from)[:]
            self.observation[robot_offset+NODE_DATA_DIM*NUM_NODES_OF_LOCAL_GRAPH_FROM\
                             :robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_FROM+NUM_NODES_OF_LOCAL_GRAPH_TO)] \
                                = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_to)[:]
            
            
            self.observation[robot_offset+self.single_obs_dim-NUM_NODES_OF_LOCAL_GRAPH_FROM-NUM_NODES_OF_LOCAL_GRAPH_TO-2:\
                             robot_offset+self.single_obs_dim-NUM_NODES_OF_LOCAL_GRAPH_TO-2] \
                                = self.gnn_nodes_from[:]
            self.observation[robot_offset+self.single_obs_dim-NUM_NODES_OF_LOCAL_GRAPH_TO-2:\
                             robot_offset+self.single_obs_dim-2] \
                                = self.gnn_nodes_to[:]
            """
            
            try:
                self.gnn_nodes_from_start[:] = self.graph.get_node_id_from(state_node, NUM_NODES_OF_LOCAL_GRAPH_FROM_START)[:]
                self.gnn_nodes_to_start[:] = self.graph.get_node_id_to(state_node, NUM_NODES_OF_LOCAL_GRAPH_TO_START+1)[1:]
                self.gnn_nodes_from_dest[:] = self.graph.get_node_id_from(dest_node, NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST)[:]
                self.gnn_nodes_to_dest[:] = self.graph.get_node_id_to(dest_node, NUM_NODES_OF_LOCAL_GRAPH_TO_DEST+1)[1:]

                self.each_observation[robot_offset:\
                                robot_offset+NODE_DATA_DIM*NUM_NODES_OF_LOCAL_GRAPH_FROM_START] \
                                    = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_from_start)[:]
                self.each_observation[robot_offset+NODE_DATA_DIM*NUM_NODES_OF_LOCAL_GRAPH_FROM_START:\
                                robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START)] \
                                    = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_to_start)[:]
                self.each_observation[robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START):\
                                robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST)] \
                                    = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_from_dest)[:]
                self.each_observation[robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST):\
                                robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST)] \
                                    = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_to_dest)[:]
                """
                nodeslen = NUM_NODES_OF_LOCAL_GRAPH_START + NUM_NODES_OF_LOCAL_GRAPH_DEST
                self.each_observation[robot_offset+NODE_DATA_DIM*nodeslen:\
                                robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_FROM_START]\
                                = self.gnn_nodes_from_start[:]
                self.each_observation[robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_FROM_START:\
                                robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START]\
                                = self.gnn_nodes_to_start[:]
                self.each_observation[robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START:\
                                robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST]\
                                = self.gnn_nodes_from_dest[:]
                self.each_observation[robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST:\
                                robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST]\
                                = self.gnn_nodes_to_dest[:]
                #"""
                
                self.each_info[robot_info_offset:\
                                robot_info_offset+NUM_NODES_OF_LOCAL_GRAPH_FROM_START]\
                                = self.gnn_nodes_from_start[:]
                self.each_info[robot_info_offset+NUM_NODES_OF_LOCAL_GRAPH_FROM_START:\
                                robot_info_offset+NUM_NODES_OF_LOCAL_GRAPH_START]\
                                = self.gnn_nodes_to_start[:]
                self.each_info[robot_info_offset+NUM_NODES_OF_LOCAL_GRAPH_START:\
                                robot_info_offset+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST]\
                                = self.gnn_nodes_from_dest[:]
                self.each_info[robot_info_offset+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST:\
                                robot_info_offset+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST]\
                                = self.gnn_nodes_to_dest[:]

                self.each_observation[robot_offset+self.single_obs_dim-STATE_ID_PRIORITY] = priority
                self.each_observation[robot_offset+self.single_obs_dim-STATE_ID_PICK_TIMER] = pick_timer
                #print(self.each_observation)
                
            except ValueError:
                print(self.gnn_nodes_from_start.shape)
                print(NUM_NODES_OF_LOCAL_GRAPH_FROM_START)
                print(self.graph.get_node_id_from(state_node, NUM_NODES_OF_LOCAL_GRAPH_FROM_START))
                raise Exception("valueerror")
            
            
            """
            self.gnn_nodes_from_start[:] = self.graph.get_node_id_from(state_node, NUM_NODES_OF_LOCAL_GRAPH_FROM_START)[:]
            self.gnn_nodes_to_start[:] = self.graph.get_node_id_to(state_node, NUM_NODES_OF_LOCAL_GRAPH_TO_START+1)[1:]
            self.gnn_nodes_from_dest[:] = self.graph.get_node_id_from(dest_node, NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST)[:]
            self.gnn_nodes_to_dest[:] = self.graph.get_node_id_to(dest_node, NUM_NODES_OF_LOCAL_GRAPH_TO_DEST+1)[1:]

            self.observation[robot_offset:\
                             robot_offset+NODE_DATA_DIM*NUM_NODES_OF_LOCAL_GRAPH_FROM_START] \
                                = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_from_start)[:]
            self.observation[robot_offset+NODE_DATA_DIM*NUM_NODES_OF_LOCAL_GRAPH_FROM_START:\
                             robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START)] \
                                = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_to_start)[:]
            self.observation[robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START):\
                             robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST)] \
                                = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_from_dest)[:]
            self.observation[robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST):\
                             robot_offset+NODE_DATA_DIM*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST)] \
                                = self.graph.get_gnn_node_data(robot_id, self.gnn_nodes_to_dest)[:]
            
            nodeslen = NUM_NODES_OF_LOCAL_GRAPH_START + NUM_NODES_OF_LOCAL_GRAPH_DEST
            self.observation[robot_offset+NODE_DATA_DIM*nodeslen:\
                             robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_FROM_START]\
                             = self.gnn_nodes_from_start[:]
            self.observation[robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_FROM_START:\
                             robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START]\
                             = self.gnn_nodes_to_start[:]
            self.observation[robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START:\
                             robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST]\
                             = self.gnn_nodes_from_dest[:]
            self.observation[robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST:\
                             robot_offset+NODE_DATA_DIM*nodeslen+NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST]\
                             = self.gnn_nodes_to_dest[:]

            self.observation[robot_offset+self.single_obs_dim-2] = priority
            self.observation[robot_offset+self.single_obs_dim-1] = pick_timer
            """            
        
        return

    def draw_data(self, img):
        completetasks = 0
        long_tasks = 0
        short_tasks = 0
        if TASK_ASSIGN_MODE and not SINGLE_TASK_MODE:
            completetasks = self.robot_envs[0].assigned_task.completetasks
            short_tasks = completetasks
        else:
            for i in range(self.num_agents):
                task_id = self.robot_envs[i].task_id
                completetasks += task_id
                num_longtask = self.robot_envs[i].tasks_is_picking[:task_id].count(True)
                long_tasks += num_longtask
                short_tasks += task_id - num_longtask
        state_text = (
            f"TStep: {self.num_steps}\n"
            #f"PColl    : {self.cnt_privent_collision}\n"
            f"PColl    : {self.cnt_all_privent_collision}\n"
            f"PCollR    : {100*self.cnt_all_privent_collision/((self.num_steps+1)*self.num_agents)}\n"
            #f"NO_OP  : {self.cnt_noop}\n"
            f"Reward  : {self.reward}\n"
            f"Short Task: {short_tasks}\n"
            f"Long Task : {long_tasks}\n"
            f"Comp Task: {completetasks}\n"
            #f"Return  : {returns}\n"
            )
        draw_text(
            img=img,
            text=state_text,
            uv_top_left=(IMAGE_WIDTH - 50, 100),
            color=(0,0,0),
            fontScale=0.75,
        )
        pass