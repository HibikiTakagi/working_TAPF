import numpy as np
import random
import cv2
import gymnasium as gym

#from numba import jit, prange
#from numba.experimental import jitclass
#from multiprocessing import Pool

from gymnasium import spaces
from graphDB import Graph, NODE_DATA_DIM

from RobotEnvironment import RobotEnvironment, draw_text
from CONST import (
    PENALTY_PRIVENT_COLLISION, NUM_AGENTS, MODE_AGENT, MAX_TIMESTEP, 
    PENALTY_NO_OP, PENALTY_PRIORITY,
    FULLNODEDATAMODE, PARTIALMODE, SINGLE_AGENT_MODE,PARTIAL_AGENT_MODE, LOCALGCNMODE, NUM_NODES_OF_LOCAL_GRAPH_TO, NUM_NODES_OF_LOCAL_GRAPH_FROM,
    KNN_MODE, KNN_AGENTS
    )
from map_maker import IS_RANDOM, LEN_NODE, IMAGE_WIDTH, ACTION_LIST, ROUTE, TEST_MODE

NUM_PROCESS=16
IS_MULTI_PROCESS = False

def unwrap_self_f(arg):
    # メソッドfをクラスメソッドとして呼び出す関数
    return MultiRobotsEnvironment.single_phase3_transition(*arg)

class MultiRobotsEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, mode_id=1):
        super(MultiRobotsEnvironment, self).__init__()
        self.mode_id = mode_id
        self.graph = Graph()

        self.num_agents = MODE_AGENT[self.mode_id]
        self.robot_envs = [RobotEnvironment(i) for i in range(self.num_agents)]

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
            self.single_obs_dim = (self.robot_envs[0].action_dim + 1) * self.nodedata_dim + 2 
            # start, route, allstart, allroute, allnumrobot
            self.obs_dim = self.single_obs_dim * self.num_agents
        elif PARTIAL_AGENT_MODE:
            self.nodedata_dim = NODE_DATA_DIM
            self.single_obs_dim = (self.robot_envs[0].action_dim + 1) * self.nodedata_dim + 2
            # start, route, partstart, partroute, partnumrobot
            self.obs_dim = self.single_obs_dim * self.num_agents
        elif LOCALGCNMODE:
            self.num_nodes_of_local_graph_from = NUM_NODES_OF_LOCAL_GRAPH_FROM
            self.num_nodes_of_local_graph_to = NUM_NODES_OF_LOCAL_GRAPH_TO
            self.nodedata_dim = NODE_DATA_DIM * self.num_nodes_of_local_graph_from


            self.single_obs_dim = (self.robot_envs[0].action_dim + 1) * self.nodedata_dim \
                                        + self.num_nodes_of_local_graph \
                                            + 2
            
            
            # start, route, partstart, partroute, partnumrobot
            self.obs_dim = self.single_obs_dim * self.num_agents

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
            else:
                self.net_state_dim = self.obs_dim
        else:
            self.net_act_dim = self.robot_envs[0].action_dim * self.num_agents
            self.net_state_dim = self.obs_dim
        #"""
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.zeros_observe = np.zeros(self.obs_dim, dtype=np.float32)
        self.observation = np.zeros(self.obs_dim, dtype=np.float32)
        self.zeros_full_observe = np.zeros(self.full_obs_dim, dtype=np.float32)
        self.full_observation = np.zeros(self.full_obs_dim, dtype=np.float32)

        self.zeros_node = np.zeros(LEN_NODE, dtype=np.float32)
        #self.nodes_start = np.zeros(LEN_NODE, dtype=np.float32)
        #self.nodes_route = np.zeros(LEN_NODE, dtype=np.float32)
        #self.nodes_num_robot = np.zeros(LEN_NODE, dtype=np.float32)

        self.reserve_node = np.zeros(LEN_NODE, dtype=np.float32)
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

        #if IS_MULTI_PROCESS:
        #    self.pool = Pool(processes=NUM_PROCESS)
        
    def reset(self):
        start_node_idxes = random.sample(list(range(LEN_NODE)), self.num_agents)
        for i in range(self.num_agents):
            self.robot_envs[i].multi_reset(state_node=start_node_idxes[i])
        self.num_steps = 0
        self.cnt_noop = 0
        self.cnt_privent_collision = 0
        self.cnt_all_privent_collision = 0
        self.rewards = np.array([0.0 for _ in range(self.num_agents)])
        self.processed_rewards = np.array([0.0 for _ in range(self.num_agents)])
        self.reward = 0.0
        
        for i in range(NUM_AGENTS):
            self.graph.register_robot_data(i, self.robot_envs[i])
        self.graph.make_nodes_common_data()

        self.make_observe()
        self.state[:] = self.observation[:]

        for i in range(NUM_AGENTS):
            state_node, _, _, _, _ = self.robot_envs[i].curr_state
            self.state_nodes[i] = state_node
        
        return self.state.copy()
    
    #@profile
    def step(self, action):
        observation, reward = self.transition(action)
        terminated = self.is_terminal
        truncated = (MAX_TIMESTEP<=self.num_steps) and (not terminated)
        info = {}
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
            self.robot_envs[i].draw_data(self.fig)
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
        num_noop_in_step = 0
        num_privent_collision_in_step = 0
        sum_violate_priority = 0.0
        
        self.reserve_node[:] = self.zeros_node[:]
        self.post_state_nodes[:] = self.zero_state_nodes[:]
        self.state_nodes[:] = self.zero_state_nodes[:]

        self.priority[:] = self.zero_priority[:]
        
        ### Phase 1 : Reserve
        for i in range(NUM_AGENTS):
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
            col_agent_idx, = np.where(self.post_state_nodes==col_node)
            col_priority = self.priority[col_agent_idx]
            if col_node in self.state_nodes:
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
            self.phase3_transition(action)
        
        self.cnt_privent_collision += num_privent_collision_in_step
        self.cnt_all_privent_collision += num_all_privent_collision_in_step
        self.cnt_noop += num_noop_in_step

        reward = self._reward_function(
            num_privent_collision_in_step,
            num_noop_in_step,
            sum_violate_priority
            )
        self.reward = reward
        
        self.make_observe()
        self.state[:] = self.observation[:]
        
        return self.state, reward
    #"""

    def phase3_transition(self, action):
        for i in range(NUM_AGENTS):
            robot = self.robot_envs[i]
            _, rw = robot.transition(action=action[i])
            self.rewards[i] = rw
    
    ## した２つは上をmultiprocessで並列化するためのコード
    @staticmethod
    def single_phase3_transition(robot, action):
        _, rw = robot.transition(action=action)
        return rw, robot
    
    #@profile
    def multi_phase3_transition(self, action):
        values = [(self.robot_envs[i], action[i]) for i in range(NUM_AGENTS)] 
        result = self.pool.map(unwrap_self_f, values, 1+NUM_AGENTS//(NUM_PROCESS))[:]
        #result = pool.map(self.single_phase3_transition, values)[:]
        self.robot_envs = [r[1] for r in result]
        self.rewards = np.array([r[0] for r in result])
        #print(self.num_steps, self.rewards)

    def _reward_function(self, col, noop, pri):
        processed_reward = (
            self.rewards[0]
            + np.sum(self.rewards[self.graph.get_knn_agents_array(0)])/KNN_AGENTS
            - col * PENALTY_PRIVENT_COLLISION
            - noop * PENALTY_NO_OP
            - pri/self.num_agents * PENALTY_PRIORITY
        ) 
        return processed_reward

    def collision_count(self, collision_nodes):
        num_all_privent_collision_in_step = 0
        for i in range(NUM_AGENTS):
            if self.post_state_nodes[i] in collision_nodes and self.post_state_nodes[i] != self.state_nodes[i]:
                num_all_privent_collision_in_step += 1
        return num_all_privent_collision_in_step

    #@profile
    def make_observe(self):
        for i in range(NUM_AGENTS):
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
            self.knn_state_next_maker(self.observation)
            self.graph.make_knn_data()
            self.knn_state_maker(self.observation)
            #if PARTIAL_AGENT_MODE:
            #    self.make_add_knninfo_observe_part_agent_node()

    def knn_state_maker(self, state):
        state = state.reshape(NUM_AGENTS, -1)
        self.replay_obs[:] = np.concatenate(state[self.graph.knn_agents[0]])
    
    def knn_state_next_maker(self, state):
        state = state.reshape(NUM_AGENTS, -1)
        #print(f"replay_obs:{self.replay_obs.shape}")
        #print(f"state:{state[self.graph.knn_agents[0]].shape}")
        self.replay_obs_next[:] = np.concatenate(state[self.graph.knn_agents[0]])

    def make_full_observation(self):
        ## unavailable
        self.full_observation[:] = self.zeros_full_observe[:]
        for i in range(self.num_agents):
            state_node, task_id, pick_timer, task_timer, priority = self.robot_envs[i].next_state
            robot_offset = i * self.robot_envs[0].obs_dim
            self.full_observation[robot_offset + state_node] = 1
            #self.full_observation[robot_offset+LEN_NODE:robot_offset+LEN_NODE*2+2] = self.robot_envs[i].state[LEN_NODE:LEN_NODE*2+2].copy()
            self.full_observation[robot_offset+LEN_NODE:robot_offset+LEN_NODE*2] = self.robot_envs[i].state[0:LEN_NODE]
            self.full_observation[robot_offset+LEN_NODE:robot_offset+LEN_NODE*2 + 1] = priority
            self.full_observation[robot_offset+LEN_NODE:robot_offset+LEN_NODE*2 + 2] = pick_timer

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
            dest_id = len(ACTION_LIST[state_node])
            for j, node in enumerate(ACTION_LIST[state_node]):
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
            dest_id = len(ACTION_LIST[state_node])
            for j, node in enumerate(ACTION_LIST[state_node]):
                node_offset = robot_offset+self.nodedata_dim*j
                self.observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, node)[:]
            node_offset = robot_offset+self.nodedata_dim*dest_id 
            self.observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, dest_node)[:]
            
            self.observation[robot_offset+self.single_obs_dim-2] = priority
            self.observation[robot_offset+self.single_obs_dim-1] = pick_timer
            
            #self.observation[robot_offset+self.single_obs_dim-2-KNN_AGENTS] = priority
            #self.observation[robot_offset+self.single_obs_dim-1-KNN_AGENTS] = pick_timer
    #'''
    def make_add_knninfo_observe_part_agent_node(self):
        for robot_id in range(self.num_agents):
            robot_offset = robot_id * self.single_obs_dim
            self.graph.get_knn_agents_distanse_array(robot_id)
            #self.observation[robot_offset+self.single_obs_dim-3-KNN_AGENTS:robot_offset+self.single_obs_dim-3] = self.graph.get_knn_agents_distanse_array(robot_id)[:]
    
    def make_observe_gcn_part_agent_node(self):
        """
        start_nodedata, 
        action_nodedata, 
        action_nodedata,
        ...
        dest_nodedata,

        node_id_FROM,
        node_id_TO,

        priority
        pick_timer
        """
        self.observation[:] = self.zeros_observe[:]
        
        for robot_id in range(self.num_agents):
            state_node, task_id, pick_timer, task_timer, priority = self.robot_envs[robot_id].next_state
            dest_node = self.robot_envs[robot_id].tasks_node[task_id]
            
            robot_offset = robot_id * self.single_obs_dim
            dest_id = len(ACTION_LIST[state_node])
            for j, node in enumerate(ACTION_LIST[state_node]):
                node_offset = robot_offset+self.nodedata_dim*j
                self.observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, node)[:]
            node_offset = robot_offset+self.nodedata_dim*dest_id 
            self.observation[node_offset:node_offset+self.nodedata_dim] = self.graph.get_node_data(robot_id, dest_node)[:]
            
            self.observation[robot_offset+self.single_obs_dim-2] = priority
            self.observation[robot_offset+self.single_obs_dim-1] = pick_timer

    def draw_data(self, img):
        completetasks = 0
        long_tasks = 0
        short_tasks = 0
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