import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collections

#from numba import jit, prange
#from numba.experimental import jitclass

import gymnasium as gym
from gymnasium import spaces
from CONST import PENALTY_COLLISION, PENALTY_NO_EXCEPTION, MAX_TIMESTEP, NUM_AGENTS, MAX_TRAIN_EPS
from map_maker import IMAGE_WIDTH, IMAGE_HEIGHT, IS_RANDOM, PICKERON, MIDDLE_PICKERON, TASK_ASSIGN_MODE, SINGLE_TASK_MODE
#from map_maker import *
from TaskManager import *

LINE_THICKNESS = 2
NODE_RADIUS = 15
ROBOT_RADIUS = 10
TARGET_RADIUS = 5
ROBOT0_RADIUS = 15
TARGET0_RADIUS = 10
#NODE_RADIUS = 30
#ROBOT_RADIUS = 15

#NODE_RADIUS = 8
#ROBOT_RADIUS = 5
#TARGET_RADIUS = 5
#ROBOT0_RADIUS = 5
#TARGET0_RADIUS = 5

# BGR colors
BLACK = (0, 0, 0)
GRAY = (240, 240, 240)
SLATEGRAY = (112, 128, 144)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
COLORLIST = [
    (66, 245, 245), (203, 245, 66), (76, 153, 0), (66, 66, 245),
    (128, 0, 128), (0, 128, 128), (128, 128, 0), (128, 128, 128),
    (0, 245, 0), (245, 0, 245), (0, 0, 245), (66, 128, 245),
    (200, 0, 200), (204, 153, 255), (255, 178, 102),
    ]

TRUNC_MODE = False

def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
    ):

    """
    Draws multiline on an image with an outline.
    """
    assert isinstance(text, str)

    # top left coordinate
    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    # split lines of the text, then draw each line
    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


class RobotEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    # def __init__(self, robot_id, len_node, connectdict, coordlist, node_distances, weights, actionlist, pickernode, is_pickup, route):
    def __init__(self, robot_id, len_node, connectdict, coordlist, node_distances, weights, actionlist, pickernode, is_pickup, assigned_task):
        super(RobotEnvironment, self).__init__()
        if TRUNC_MODE:
            self.robot_id = 0
        else:
            self.robot_id = robot_id
        self.eps = 0
        self.map_init_reset(len_node, connectdict, coordlist, node_distances, weights, actionlist, pickernode, is_pickup, assigned_task)

    def map_init_reset(self, len_node, connectdict, coordlist, node_distances, weights, actionlist, pickernode, is_pickup, assigned_task):
        #print("RobotEnvironment.pyのmap_init_resetが呼び出されました。")
        #self.obs_dim = LEN_NODE * 2 + 2
        self.len_node = len_node
        self.connect_dict = connectdict
        self.coord_list = coordlist
        self.node_distances = node_distances
        self.weights = weights
        self.actionlist = actionlist
        self.picker_node = pickernode
        self.assigned_task = assigned_task
        self.is_pickup =is_pickup

        self.obs_dim = self.len_node
        #"""
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.observe = np.zeros(self.obs_dim, dtype=np.float32)
        self.tmp_observe = np.zeros(self.obs_dim, dtype=np.float32)
        self.zeros_observe = np.zeros(self.obs_dim, dtype=np.float32)
        #"""
        self.observation_dim = (self.obs_dim,)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_dim, dtype=np.float32)
        self.act_dim = max(len(num_connect)+1 for num_connect in self.connect_dict.values())
        self.action_dim = self.act_dim
        self.action_space = gym.spaces.Discrete(self.action_dim)
        #self.action_dim = (self.act_dim,)
        #self.action_space = gym.spaces.Box(low=0, high=1, shape=self.action_dim, dtype=np.float32)
        self.reward_range = (-float('inf'), float('inf'))
        self.num_steps = None
        
        self.priority = random.random()
        self.task_id = None

        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                self.tasks_node = self.assigned_task[self.robot_id]
                self.tasks_is_picking = self.is_pickup[self.robot_id]
            else:
                #self.task_node = self.assigned_task.task_random_assign(self.robot_id)
                self.task_node = self.assigned_task.reserve_next_task_random(self.robot_id) #この段階ではstate_nodeが割り当てられていない。
                self.assigned_task.reserved_task_assign(self.robot_id)
                self.tasks_is_picking = [False for _ in range(500)] # 500ではなく、本当はタスク列の長さが良い
        else:
            if IS_RANDOM:
                self.tasks_node = [random.randrange(0, self.len_node) for _ in range(MAX_TIMESTEP)]
                self.tasks_is_picking = [random.choice([True, False]) for _ in range(len(self.tasks_node))]
            else:
                #self.tasks_node = self.route[self.robot_id]
                self.tasks_node = self.assigned_task[self.robot_id]
                self.tasks_is_picking = self.is_pickup[self.robot_id]

        self.pick_timer = None
        self.task_timer = None
        self.log_reward = 0
        
        self.fig = None
        self.ax = None
        """
        if self.robot_id==0:
            self.color = COLORLIST[self.robot_id%len(COLORLIST)]
        else:
            self.color = BLACK
        """
        self.color = COLORLIST[self.robot_id%len(COLORLIST)]
        #"""
        self.zeros_node = np.zeros(self.len_node, dtype=np.float32)
        self.distance_array = np.zeros(self.len_node, dtype=np.uint16)
        self.route_mindist_array = np.zeros(self.len_node, dtype=np.uint16)
        self.not_min_route_mask = np.zeros(self.len_node, dtype=np.bool_)
        self.min_route_mask = np.zeros(self.len_node, dtype=np.bool_) # use multi
        self.cnt_data = np.zeros(self.len_node, dtype=np.float32)
        #"""
    
    def commonreset(self):
        #print("RobotEnvironmentのcommonresetが呼び出されました。")
        self.num_steps = 0
        self.priority = random.random()
        self.task_id = 0
        self.task_timer = 1
        self.change_destination = False
        self.wrong_op_cnt = 0
        self.wrong_op = False
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                self.pick_timer = self.weights[self.tasks_node[self.task_id]] if self.tasks_is_picking[0] else 0
                self.nextdist = 0
                self.curr_state = [self.state_node, self.task_id, self.pick_timer, self.task_timer, self.priority]
                self.next_state = [self.state_node, self.task_id, self.pick_timer, self.task_timer, self.priority]
            else:
                self.pick_timer = self.weights[self.task_node] if self.tasks_is_picking[0] else 0
                self.nextdist = self.node_distances[self.state_node, self.assigned_task.reserve_next_task(self.robot_id, self.state_node)]
                self.curr_state = [self.state_node, self.task_node, self.pick_timer, self.task_timer, self.priority]
                self.next_state = [self.state_node, self.task_node, self.pick_timer, self.task_timer, self.priority]
        else:
            self.pick_timer = self.weights[self.tasks_node[self.task_id]] if self.tasks_is_picking[0] else 0 
            self.distance_between_task_list = [self.node_distances[self.tasks_node[i], self.tasks_node[i+1]] for i in range(len(self.tasks_node)-1)]
            self.distance_between_task_list.insert(0,0)
            self.nextdist = self.node_distances[self.state_node, self.tasks_node[self.task_id+1]]
            self.curr_state = [self.state_node, self.task_id, self.pick_timer, self.task_timer, self.priority]
            self.next_state = [self.state_node, self.task_id, self.pick_timer, self.task_timer, self.priority]
        self.log_reward = 0
        
        #self.state = empty
        self.make_observe(self.next_state)
        self.state[:] = self.observe[:]

    def reset(self):
        self.eps += 1
        if IS_RANDOM:
            self.make_new_tasks()
        self.state_node = self.tasks_node[0]
        self.commonreset()
        
        return self.state
    
    def multi_reset(self, state_node:int):
        #print("RobotEnvironmentのmulti_resetが呼び出されました。")
        self.eps += 1
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                self.state_node = self.assigned_task[self.robot_id + NUM_AGENTS][0]
            else:
                self.state_node = self.task_node
        else:
            if IS_RANDOM:
                self.make_new_tasks(state_node)
                self.state_node = state_node
            else:
                self.state_node = self.tasks_node[0]
        self.commonreset()
        return
    
    def step(self, actions):
        action = actions
        #action = np.argmax(actions)
        observation, reward = self.transition(action)
        terminated = self.is_terminal
        truncated =  (MAX_TIMESTEP<=self.num_steps) and (not terminated)
        info = {}
        self.num_steps += 1
        done = terminated or truncated
        if TRUNC_MODE:
            return observation, reward, done, info
        else:
            return observation, reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        self.fig = self.draw_graph()
        self.draw_robot(self.fig)
        self.draw_data(self.fig)
        self.fig = cv2.cvtColor(self.fig, cv2.COLOR_BGR2RGB)
        return self.fig
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        pass
    
    # 以下追加
    
    @property
    def is_terminal(self):
        return False

    @property
    def dest_node(self):
        return self.tasks_node[self.task_id]
    
    @property
    def is_picking(self):
        return self.curr_state[2] > 0
    
    @property
    def distance_to_next_task(self):
        return self.distance_between_task_list[self.task_id]
    
    #@jit(nopython=False)
    #@profile # if GNN  ignore time
    def transition(self, action):
        #print("RobotEnvironmentのtransitionが呼び出されました。")
        self.next_state[:] = self.lookahead(action)
        reward = self.reward_function(action)
     
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                next_state_node, next_task_id, next_pick_timer, next_task_timer, next_priority = self.next_state
                #next_dest_node = self.tasks_node[min(next_task_id+1, len(self.tasks_node)-1)]
                self.state_node = self.next_state[0]
                self.task_id    = self.next_state[1]
                self.pick_timer = self.next_state[2]
                self.task_timer = self.next_state[3]
            else:
                next_state_node, next_task_node, next_pick_timer, next_task_timer, next_priority = self.next_state
                #next_dest_node = self.assigned_task.virtual_task_assign()
                if next_state_node == next_task_node and next_pick_timer == 0:
                    self.assigned_task.update_tasks(self.robot_id)
                    self.next_state[1] = self.assigned_task.reserved_task_assign(self.robot_id)
                    self.assigned_task.reserve_next_task(self.robot_id, self.state_node)
                next_dest_node = self.assigned_task.reserved_task_checker[self.robot_id]
                self.state_node = self.next_state[0]
                self.task_node  = self.next_state[1]
                self.pick_timer = self.next_state[2]
                self.task_timer = self.next_state[3] 
        else:
            next_state_node, next_task_id, next_pick_timer, next_task_timer, next_priority = self.next_state
            next_dest_node = self.tasks_node[min(next_task_id+1, len(self.tasks_node)-1)]
            self.state_node = self.next_state[0]
            self.task_id    = self.next_state[1]
            self.pick_timer = self.next_state[2]
            self.task_timer = self.next_state[3]

        if self.wrong_op:
            self.wrong_op_cnt += 1
        if self.change_destination:
            self.wrong_op_cnt = 0
            self.nextdist = self.node_distances[next_state_node, next_dest_node]
            self.change_destination = False
        #print(f"RobotEnvironmentのtransitionの結果、エージェント{self.robot_id}のstate_nodeは{self.state_node}で、task_nodeは{self.task_node}となりました。")
        #self.next_state[4] = random.random()/2
        #if self.curr_state[0] == self.next_state[0]:
        #    self.next_state[4] = 1.0
        self.priority = self.next_state[4]
        
        self.curr_state[:] = self.next_state[:]
        self.make_observe(self.next_state) # GNN 85.0
        self.state[:] = self.observe[:]
        self.log_reward = reward
        return self.state, reward
    
    def virtual_transition(self, action):

        #self.next_state[:] = self.lookahead(action)
        virtual_next_state = self.lookahead(action)
        reward = self.reward_function(action)
        
        next_state_node, next_task_id, next_pick_timer, next_task_timer, next_priority = self.next_state
        next_dest_node = self.tasks_node[min(next_task_id+1, len(self.tasks_node)-1)]
        
        if self.wrong_op:
            self.wrong_op_cnt += 1
        if self.change_destination:
            self.wrong_op_cnt = 0
            self.nextdist = self.node_distances[next_state_node, next_dest_node]
            self.change_destination = False
        
        #self.state_node = self.next_state[0]
        #self.task_id    = self.next_state[1]
        #self.pick_timer = self.next_state[2]
        #self.task_timer = self.next_state[3]
        
        #self.priority = self.next_state[4]
        
        #self.curr_state[:] = self.next_state[:]
        virtual_observe = self.make_virtual_observe(virtual_next_state) # GNN 85.0
        
        return virtual_observe, reward

    #@profile
    def pretransition_state(self, action):
        self.next_state[:] = self.lookahead(action)
        #return self.next_state[0], self.next_state[4]
        return self.next_state[0]

    def reward_function(self, action):
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                next_state_node, next_task_id, next_pick_timer, next_task_timer, next_priority = self.next_state
                state_node, task_id, pick_timer, task_timer, curr_priority = self.curr_state
                newdist = self.node_distances[next_state_node, self.tasks_node[next_task_id]]
                olddist = self.node_distances[state_node, self.tasks_node[next_task_id]]
                if newdist >= olddist and pick_timer == 0:
                    self.wrong_op = True
                else:
                    self.wrong_op = False
                if pick_timer>0:
                    return 1
                elif next_state_node == self.tasks_node[next_task_id]:
                    self.change_destination = True
                    #print(f"{next_task_id } {self.nextdist}  {task_timer}")
                    return self.nextdist/(task_timer+1)
                else:
                    if self.wrong_op:
                        #return -1/(newdist+1)# 20240514 change
                        return max(-1, -(self.wrong_op_cnt+1)/(self.nextdist+1))
                    else:
                        #return 0 # 20240514 change
                        return max(-1, -(self.wrong_op_cnt)/(self.nextdist+1))
            else:
                next_state_node, next_task_node, next_pick_timer, next_task_timer, next_priority = self.next_state
                state_node, task_node, pick_timer, task_timer, curr_priority = self.curr_state
                newdist = self.node_distances[next_state_node, next_task_node]
                olddist = self.node_distances[state_node, next_task_node]
                # print("next_task_node!!", next_task_node)
                if newdist >= olddist and pick_timer == 0:
                    self.wrong_op = True
                else:
                    self.wrong_op = False
                if pick_timer>0:
                    return 1
                elif next_state_node == next_task_node:
                    self.change_destination = True
                    #print(f"{next_task_id } {self.nextdist}  {task_timer}")
                    return self.nextdist/(task_timer+1)
                else:
                    if self.wrong_op:
                        #return -1/(newdist+1)# 20240514 change
                        return max(-1, -(self.wrong_op_cnt+1)/(self.nextdist+1))
                    else:
                        #return 0 # 20240514 change
                        return max(-1, -(self.wrong_op_cnt)/(self.nextdist+1))
        else:
            next_state_node, next_task_id, next_pick_timer, next_task_timer, next_priority = self.next_state
            state_node, task_id, pick_timer, task_timer, curr_priority = self.curr_state
            newdist = self.node_distances[next_state_node, self.tasks_node[next_task_id]]
            olddist = self.node_distances[state_node, self.tasks_node[next_task_id]]
            if newdist >= olddist and pick_timer == 0:
                self.wrong_op = True
            else:
                self.wrong_op = False
            if pick_timer>0:
                return 1
            elif next_state_node == self.tasks_node[next_task_id]:
                self.change_destination = True
                #print(f"{next_task_id } {self.nextdist}  {task_timer}")
                return self.nextdist/(task_timer+1)
            else:
                if self.wrong_op:
                    #return -1/(newdist+1)# 20240514 change
                    return max(-1, -(self.wrong_op_cnt+1)/(self.nextdist+1))
                else:
                    #return 0 # 20240514 change
                    return max(-1, -(self.wrong_op_cnt)/(self.nextdist+1))

    #@profile
    def lookahead(self, action):
        #print("RobotEnvironmentのlookaheadが呼び出されました。")
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                new_node, new_task_id, new_pick_timer, new_task_timer, new_priority = self.curr_state
                curr_node, curr_task_id, curr_pick_timer, curr_task_timer, curr_priority = self.curr_state
                new_priority = 1.0
                if self.is_terminal:
                    return [new_node, new_task_id, new_pick_timer, new_task_timer, new_priority]
                if curr_pick_timer>0:
                    new_pick_timer = curr_pick_timer - 1
                    new_task_timer = curr_task_timer + 1
                    return [new_node, new_task_id, new_pick_timer, new_task_timer, new_priority]
                if curr_node == self.tasks_node[self.task_id]:
                    new_task_id = min(curr_task_id + 1, len(self.tasks_node) - 1)
                    new_task_timer = 1
                else:
                    new_task_id = curr_task_id
                    new_task_timer = curr_task_timer + 1
                
                new_node = self.actionlist[curr_node][action]
                
                if new_node != curr_node:
                    new_priority = random.random()/2
                #print(f"curr_node{curr_node}>{new_node} {new_priority}")

                new_dest = self.tasks_node[new_task_id]
                new_pick_timer = 0
                if new_node == new_dest and self.tasks_is_picking[new_task_id]:
                    new_pick_timer = self.weights[new_dest]
                return [new_node, new_task_id, new_pick_timer, new_task_timer, new_priority]    
            else:
                new_node, new_task_node, new_pick_timer, new_task_timer, new_priority = self.curr_state
                curr_node, curr_task_node, curr_pick_timer, curr_task_timer, curr_priority = self.curr_state
                new_priority = 1.0
                if self.is_terminal:
                    return [new_node, new_task_node, new_pick_timer, new_task_timer, new_priority]
                if curr_pick_timer>0:
                    new_pick_timer = curr_pick_timer - 1
                    new_task_timer = curr_task_timer + 1
                    return [new_node, new_task_node, new_pick_timer, new_task_timer, new_priority]
                ### curr_pick_timer = 0 の場合 ###
                if curr_node == self.task_node:
                    new_task_node  = self.assigned_task.reserve_next_task(self.robot_id, self.state_node)
                    self.task_node = new_task_node
                    new_task_timer = 1
                else:
                    new_task_node = curr_task_node
                    new_task_timer = curr_task_timer + 1
                
                new_node = self.actionlist[curr_node][action]
                
                if new_node != curr_node:
                    new_priority = random.random()/2
                #print(f"curr_node{curr_node}>{new_node} {new_priority}")

                new_dest = self.task_node
                new_pick_timer = 0
                if new_node == new_dest and self.tasks_is_picking[0]: # idで渡せないので、とりあえず0にしている
                    new_pick_timer = self.weights[new_dest]
                return [new_node, new_task_node, new_pick_timer, new_task_timer, new_priority]
        else:
            new_node, new_task_id, new_pick_timer, new_task_timer, new_priority = self.curr_state
            curr_node, curr_task_id, curr_pick_timer, curr_task_timer, curr_priority = self.curr_state
            new_priority = 1.0
            if self.is_terminal:
                return [new_node, new_task_id, new_pick_timer, new_task_timer, new_priority]
            if curr_pick_timer>0:
                new_pick_timer = curr_pick_timer - 1
                new_task_timer = curr_task_timer + 1
                return [new_node, new_task_id, new_pick_timer, new_task_timer, new_priority]
            
            if curr_node == self.tasks_node[self.task_id]:
                new_task_id = min(curr_task_id + 1, len(self.tasks_node) - 1)
                new_task_timer = 1
            else:
                new_task_id = curr_task_id
                new_task_timer = curr_task_timer + 1
            
            new_node = self.actionlist[curr_node][action]
            
            if new_node != curr_node:
                new_priority = random.random()/2
            #print(f"curr_node{curr_node}>{new_node} {new_priority}")

            new_dest = self.tasks_node[new_task_id]
            new_pick_timer = 0
            if new_node == new_dest and self.tasks_is_picking[new_task_id]:
                new_pick_timer = self.weights[new_dest]
            return [new_node, new_task_id, new_pick_timer, new_task_timer, new_priority]

    #"""
    def make_new_tasks(self, state_node):
        for i in range(len(self.tasks_node)):
            self.tasks_node[i] = random.randrange(0,self.len_node-1)
            if PICKERON and i%2==1:
                self.tasks_node[i] = random.randrange(self.picker_node, self.len_node-1)
            if MIDDLE_PICKERON and i%2==1 and self.eps>=MAX_TRAIN_EPS//2:
                self.tasks_node[i] = random.randrange(self.picker_node, self.len_node-1)

            self.tasks_is_picking[i] = random.choice([True, False])
        self.tasks_is_picking[0] = False
        self.tasks_node[0] = state_node
    #"""

    #@profile
    def make_observe(self, state): # 85% of self.transition GNN
        # print("RobotEnvironmentのmake_observeが呼び出されました。")
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                state_node, task_id, pick_timer, task_timer, priority = state
            else:
                state_node, task_node, pick_timer, task_timer, priority = state
        else:
            state_node, task_id, pick_timer, task_timer, priority = state
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                dest_node = self.tasks_node[task_id]
            else:
                dest_node = task_node
        else:
            dest_node = self.tasks_node[task_id]
        if state_node == dest_node and pick_timer==0:
            if TASK_ASSIGN_MODE and not SINGLE_TASK_MODE:
                self.assigned_task.cancel_reserved_task(self.robot_id)
                dest_node = self.assigned_task.reserve_next_task(self.robot_id, self.state_node)
            elif not TASK_ASSIGN_MODE:
                task_id = task_id + 1
                dest_node = self.tasks_node[task_id]

        self.tmp_observe[:] = self.zeros_observe[:]
        #self.tmp_observe[state_node] = 1
        self.make_route_data(state_node, dest_node)
        self.tmp_observe[0:self.len_node]=self.cnt_data[:]
        self.observe[:] = self.tmp_observe[:]
    
    def make_virtual_observe(self, state): # 85% of self.transition GNN
        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                state_node, task_id, pick_timer, task_timer, priority = state
                dest_node = self.tasks_node[task_id]
            else:
                state_node, task_node, pick_timer, task_timer, priority = state
                dest_node = self.task_node
        else:
            state_node, task_id, pick_timer, task_timer, priority = state
            dest_node = self.tasks_node[task_id]
        if state_node == dest_node and pick_timer==0:
            task_id = task_id + 1
            dest_node = self.tasks_node[task_id]

        self.tmp_observe[:] = self.zeros_observe[:]
        #self.tmp_observe[state_node] = 1
        self.make_virtual_route_data(state_node, dest_node)
        self.tmp_observe[0:self.len_node]=self.cnt_data[:]
        return self.tmp_observe[:]
    
    #@profile
    def make_route_data(self, start_node, task_node):
        self.distance_array[:] = self.node_distances[start_node]
        self.route_mindist_array[:] = self.node_distances[start_node] + self.node_distances[:,task_node]
        
        self.not_min_route_mask[:] = self.route_mindist_array != self.node_distances[start_node, task_node]
        self.min_route_mask[:]     = self.route_mindist_array == self.node_distances[start_node, task_node]
        self.distance_array[self.not_min_route_mask] = self.len_node*2
        dist_cnt = collections.Counter(self.distance_array)
        self.cnt_data[:] = 1/np.array([dist_cnt[key] for key in self.distance_array])[:]
        self.cnt_data[self.not_min_route_mask] = 0.0
        
        self.cnt_data[start_node] = 0.0 # 20240501 add(入れなければ1.0:最短経路上のため)
    
    #@profile
    def make_virtual_route_data(self, start_node, task_node):
        distance_array = self.node_distances[start_node]
        route_mindist_array = self.node_distances[start_node] + self.node_distances[:,task_node]
        
        not_min_route_mask = route_mindist_array != self.node_distances[start_node, task_node]
        min_route_mask     = route_mindist_array == self.node_distances[start_node, task_node]
        distance_array[self.not_min_route_mask] = self.len_node*2
        dist_cnt = collections.Counter(self.distance_array)
        cnt_data = 1/np.array([dist_cnt[key] for key in distance_array])[:]
        cnt_data[self.not_min_route_mask] = 0.0
        
        cnt_data[start_node] = 0.0 # 20240501 add(入れなければ1.0:最短経路上のため)

    def draw_graph_uav(self):
        graph_frame_height = IMAGE_HEIGHT
        graph_frame_width = IMAGE_WIDTH

        # reserve space for textual info (just padding)
        frame_height = graph_frame_height + 200
        frame_width = graph_frame_width + 200

        frame_height = graph_frame_height + 400
        frame_width = graph_frame_width + 200

        img = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        
        for k, vs in self.connect_dict.items():
            point1 = self.coord_list[k]
            for v in vs:
                point2 = self.coord_list[v]
                cx, cy = point2
                vx, vy = point1
                a = vy - cy
                b = vx - cx
                xa = int((cx - b*NODE_RADIUS/((a**2+b**2)**(1/2))))
                ya = int((cy + a*NODE_RADIUS/((a**2+b**2)**(1/2))))
                xb = int((cx + b*NODE_RADIUS/((a**2+b**2)**(1/2))))
                yb = int((cy - a*NODE_RADIUS/((a**2+b**2)**(1/2))))
                if (vx-xa)**2+(vy-ya)**2 < (vx-xb)**2+(vy-yb)**2:
                    pointa = (xa,ya)
                else:
                    pointa = (xb,yb)
                cv2.arrowedLine(img, point1, pointa, color=BLACK, thickness=LINE_THICKNESS,tipLength=0.2)
        
        for node in self.connect_dict:
            coord = self.coord_list[node]
            #### UAV4-mode ####
            len_edge = 4
            if (node%(len_edge*len_edge)) in [0, 2, 5, 7, 8, 10, 13, 15]:
                cv2.circle(img, coord, radius=NODE_RADIUS*2, color=BLUE, thickness=cv2.FILLED)
            else:
                cv2.circle(img, coord, radius=NODE_RADIUS*2, color=RED, thickness=cv2.FILLED)
            #### AGV-mode ###
            #cv2.circle(img, coord, radius=NODE_RADIUS, color=BLUE, thickness=cv2.FILLED)
            
            cv2.putText(
                img,
                text=str(node),
                org=(coord[0] - 15, coord[1] + 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=WHITE,
                thickness=2,
            )
        
        #self.fig = img
        return img
    
    def draw_graph(self):
        graph_frame_height = IMAGE_HEIGHT
        graph_frame_width = IMAGE_WIDTH

        # reserve space for textual info (just padding)
        #frame_height = graph_frame_height + 200
        #frame_width = graph_frame_width + 200

        frame_height = graph_frame_height + 0
        frame_width = graph_frame_width + 200

        img = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        
        for k, vs in self.connect_dict.items():
            point1 = self.coord_list[k]
            for v in vs:
                point2 = self.coord_list[v]
                cx, cy = point2
                vx, vy = point1
                a = vy - cy
                b = vx - cx
                xa = int((cx - b*NODE_RADIUS/((a**2+b**2)**(1/2))))
                ya = int((cy + a*NODE_RADIUS/((a**2+b**2)**(1/2))))
                xb = int((cx + b*NODE_RADIUS/((a**2+b**2)**(1/2))))
                yb = int((cy - a*NODE_RADIUS/((a**2+b**2)**(1/2))))
                if (vx-xa)**2+(vy-ya)**2 < (vx-xb)**2+(vy-yb)**2:
                    pointa = (xa,ya)
                else:
                    pointa = (xb,yb)
                cv2.arrowedLine(img, point1, pointa, color=BLACK, thickness=LINE_THICKNESS,tipLength=0.2)
        
        for node in self.connect_dict:
            coord = self.coord_list[node]
            #### AGV-mode ###
            #cv2.circle(img, coord, radius=NODE_RADIUS, color=BLUE, thickness=cv2.FILLED)
            cv2.circle(img, coord, radius=NODE_RADIUS, color=WHITE, thickness=cv2.FILLED)
            cv2.circle(img, coord, radius=NODE_RADIUS, color=BLACK, thickness=2)

            if node >= 100:
                org = (coord[0] - 15, coord[1] + 5)
            elif node < 10:
                org = (coord[0] - 5, coord[1] + 5)
            else:
                org = (coord[0] - 10, coord[1] + 5)
            
            cv2.putText(
                img,
                text=str(node),
                #org=(coord[0] - 15, coord[1] + 5),
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                #color=WHITE,
                color=SLATEGRAY,
                thickness=2,
            )
        
        #self.fig = img
        return img
    
    def draw_robot(self, img):
        ### destination ###
        if TASK_ASSIGN_MODE:
            if not SINGLE_TASK_MODE:
                dest_y, dest_x = self.coord_list[self.task_node]
        else:
            dest_y, dest_x = self.coord_list[self.dest_node]
        #coord = (dest_y+10, dest_x-1*((self.robot_id-1)%3))
        coord = (dest_y, dest_x)
        """
        if self.robot_id==0:
            cv2.circle(img, coord, TARGET0_RADIUS, color=self.color, thickness=3)
        else:
            cv2.circle(img, coord, TARGET_RADIUS, color=self.color, thickness=3)
        """
        cv2.circle(img, coord, NODE_RADIUS-2, color=self.color, thickness=2)

        ### agent ###
        pos_y, pos_x = self.coord_list[self.state_node]
        coord = (pos_y-10, pos_x+1*((self.robot_id-1)%3))
        if not self.is_terminal:
            cv2.circle(img, coord, ROBOT_RADIUS, color=self.color, thickness=cv2.FILLED)
            
            cv2.putText(
                img,
                text=str(self.robot_id),
                org=(coord[0]-10, coord[1]-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                #color=BLACK,
                color=RED,
                thickness=2,
            )
            
        else:
            if self.robot_id==0:
                cv2.circle(img, coord, ROBOT0_RADIUS, color=self.color, thickness=3)
            else:
                cv2.circle(img, coord, ROBOT_RADIUS, color=self.color, thickness=3)
            cv2.putText(
                img,
                text=str(self.robot_id),
                org=(coord[0]-10, coord[1]-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                #color=BLACK,
                color=RED,
                thickness=2,
            )
        
    def draw_robot_uav(self, img):
        ### destination ###
        dest_y, dest_x = self.coord_list[self.dest_node]
        coord = (dest_y+10, dest_x-1*((self.robot_id-1)%3))
        if self.robot_id==0:
            cv2.circle(img, coord, TARGET0_RADIUS*2, color=self.color, thickness=3)
        else:
            cv2.circle(img, coord, TARGET_RADIUS*2, color=self.color, thickness=3)
        
        ### agent ###
        pos_y, pos_x = self.coord_list[self.state_node]
        coord = (pos_y-10, pos_x+1*((self.robot_id-1)%3))
        if not self.is_terminal:
            cv2.circle(img, coord, ROBOT_RADIUS*2, color=self.color, thickness=cv2.FILLED)
            cv2.putText(
                img,
                text=str(self.robot_id),
                org=(coord[0]-10, coord[1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=BLACK,
                thickness=2,
            )
        else:
            if self.robot_id==0:
                cv2.circle(img, coord, ROBOT0_RADIUS*2, color=self.color, thickness=3)
            else:
                cv2.circle(img, coord, ROBOT_RADIUS*2, color=self.color, thickness=3)
            cv2.putText(
                img,
                text=str(self.robot_id),
                org=(coord[0]-10, coord[1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.50,
                color=BLACK,
                thickness=2,
            )
    
    def draw_data(self, img):
        pick = "F"
        if self.is_picking:
            pick = "T"
        change_dist = "F"
        if self.change_destination:
            change_dist = "T"

        if TASK_ASSIGN_MODE:
            if SINGLE_TASK_MODE:
                robot_text = (
                    f"R{str(self.robot_id)}:P:{self.priority:.3f}\n"
                    f"- S>D:{self.state_node}>{self.dest_node}{pick}\n"
                    f"- Timer P:{self.pick_timer} T:{self.task_timer}\n"
                    f"- TASK:{self.task_id} CD:{change_dist}\n"
                    f"- wcnt:{self.wrong_op_cnt} ND:{self.nextdist}\n"
                #    f"- ND2:{self.distance_to_next_task}\n"
                    f"- rw:{self.log_reward:.3f}\n"
                )
            else:
                robot_text = (
                    f"R{str(self.robot_id)}:P:{self.priority:.3f}\n"
                    # f"- S>D:{self.state_node}>{self.dest_node}{pick}\n"
                    f"- S>D:{self.state_node}>{self.task_node}{pick}\n"
                    f"- Timer P:{self.pick_timer} T:{self.task_timer}\n"
                    f"- TASK:{self.task_node} CD:{change_dist}\n"
                    f"- wcnt:{self.wrong_op_cnt} ND:{self.nextdist}\n"
                #    f"- ND2:{self.distance_to_next_task}\n"
                    f"- rw:{self.log_reward:.3f}\n"
                )
        else:
            robot_text = (
                f"R{str(self.robot_id)}:P:{self.priority:.3f}\n"
                f"- S>D:{self.state_node}>{self.dest_node}{pick}\n"
                f"- Timer P:{self.pick_timer} T:{self.task_timer}\n"
                f"- TASK:{self.task_id} CD:{change_dist}\n"
                f"- ND2:{self.distance_to_next_task}\n"
                f"- rw:{self.log_reward:.3f}\n"
            )
        numrow = 6
        y = self.robot_id // numrow
        x = self.robot_id % numrow
        top_left = (int(0.05 * (IMAGE_WIDTH+200) + x * 150), IMAGE_HEIGHT+5 + 120 * y)
        
        top_left = (IMAGE_WIDTH - 50, 400)
        draw_text(img, text=robot_text, uv_top_left=top_left, color=self.color, fontScale=0.75)
        #draw_text(img, text=robot_text, uv_top_left=top_left, color=self.color)
