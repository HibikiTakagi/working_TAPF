import numpy as np
from MultiRobotsEnvironment import MultiRobotsEnvironment
#from map_maker import  NODE_DISTANCES, ACTION_LIST

class ClassicPolicy:
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

