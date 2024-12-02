import sys
import os
import pickle
import logging
import cv2


import numpy as np
import matplotlib.pyplot as plt
from MultiRobotsEnvironment import MultiRobotsEnvironment
from DQL import DeepQPolicy
#from JSAnimation.IPython_display import display_animation
from matplotlib import animation
#from IPython.display import display
from tqdm import tqdm

from CONST import  (
    EXE_TIMESTEP, GAMMA, COMMONDIRSLA, DIRNAME, DIR_DICT, MODE_AGENT, TMPDIRNAME, RESULT_MOVE_FILE, DIRFRAME, NUM_AGENTS, MODE_ID,
    NUM_NODES_OF_LOCAL_GRAPH_START, NUM_NODES_OF_LOCAL_GRAPH_DEST,KNN_AGENTS
)
from map_maker import map_maker
from graphDB import NODE_INFO_ROUTE, NODE_DATA_DIM, NODE_DATA_EACH_DIM

FRAME_SAVE_ON = False
SAVE_FLAG = True
if SAVE_FLAG:
    CHECK_LOOP_NUM = 1000
else:
    CHECK_LOOP_NUM = 1

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    plt.close()

    #def animate(i):
    #    patch.set_data(frames[i])

    #anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
    #                               interval=50)

    #anim.save('movie_cartpole_DQN.mp4')  # 動画のファイル名と保存です

class Testing:
    def __init__(self, mode_id):
        self.mode_id = mode_id
        self.dirname = COMMONDIRSLA + DIRNAME + DIR_DICT[self.mode_id]
        self.dirnamesla = self.dirname + "/"
        self.tmpdirname = self.dirnamesla + TMPDIRNAME
        self.tmpdirnamesla = self.tmpdirname + "/"
        self.dirframe = self.dirnamesla+DIRFRAME
        self.diraverage = self.dirnamesla + "average"
        self.diraveragesla = self.diraverage + "/"
        self.set_directry()
        
        self.nooplist = []
        self.pcolllist = []
        self.longtasklist = []
        self.shorttasklist = []
        self.frame_flag = True
        self.env = MultiRobotsEnvironment(self.mode_id)

    def set_directry(self):
        self.resultmove_file = self.dirnamesla + RESULT_MOVE_FILE

    def append_task(self):
        completetasks = 0
        long_tasks = 0
        short_tasks = 0
        for i in range(self.env.num_agents):
            task_id = self.env.robot_envs[i].task_id
            completetasks += task_id
            num_longtask = self.env.robot_envs[i].tasks_is_picking[:task_id].count(True)
            long_tasks += num_longtask
            short_tasks += task_id - num_longtask
        self.shorttasklist.append(short_tasks)
        self.longtasklist.append(long_tasks)
    
    def save_list_txt(self, arglist, paths):
        f = open(paths,'w')
        for i in range(len(arglist)):
            f.write(f"{arglist[i]}\n")
        f.write("avg\n")
        f.write(f"{np.mean(arglist)}\n")
        f.close()

    def get_nodes(self, state, info):
        state_reshape = np.reshape(state, (NUM_AGENTS, KNN_AGENTS, -1))
        start_node_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_START
        dest_node_dims = NODE_DATA_DIM * NUM_NODES_OF_LOCAL_GRAPH_DEST
        from_nodes = NUM_NODES_OF_LOCAL_GRAPH_START
        to_nodes = NUM_NODES_OF_LOCAL_GRAPH_DEST
        elsedim = 2
        
        #split_points = np.cumsum([startnode_dims, destnode_dims, fromnodes, tonodes])
        #start_info, dest_info, state_start, state_dest,_ = np.split(state_reshape, split_points, axis=2)
        split_points = np.cumsum([start_node_dims, dest_node_dims])
        start_info, dest_info, _ = np.split(state_reshape, split_points, axis=2)
        
        graphs = info.reshape(NUM_AGENTS, KNN_AGENTS, -1)
        state_start, state_dest = np.split(graphs, np.cumsum([from_nodes]), axis=2)
        #graphs = np.concatenate([state_start, state_dest], axis=2)
        
        graphs = np.reshape(graphs, (NUM_AGENTS, -1))
        
        abst_array = np.concatenate([state_start[:, :, 0:len(self.env.actionlist[0])], state_dest[:, :, 0:1]], axis=2)
        abst_array = np.reshape(abst_array, (NUM_AGENTS, -1))
        
        
        x_start_infos = np.reshape(start_info, (self.env.num_agents, KNN_AGENTS, from_nodes, -1))
        x_dest_infos = np.reshape(dest_info, (self.env.num_agents, KNN_AGENTS, to_nodes, -1))
        #x_all_start_infos = x_start_infos[:, :,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM]
        #x_all_dest_infos = x_dest_infos[:,:,:,NODE_DATA_EACH_DIM:NODE_DATA_DIM]
        #nodes_info_gnn =  np.reshape(np.concatenate([x_all_start_infos, x_all_dest_infos], axis=2), (self.env.num_agents, KNN_AGENTS*(NUM_NODES_OF_LOCAL_GRAPH_START+NUM_NODES_OF_LOCAL_GRAPH_DEST), -1))


        x_each_start_infos = x_start_infos[:, :,:,0:NODE_DATA_EACH_DIM]
        x_each_dest_infos = x_dest_infos[:, :,:,0:NODE_DATA_EACH_DIM]
        pre_each_info = np.concatenate([x_each_start_infos, x_each_dest_infos], axis=2)

        return graphs, abst_array,  pre_each_info#, nodes_info_gnn
    
    #@profile # GNN 13% of savetraining.py
    def make_localedge_numpy(self, nodes_numpy ,connect):
        node_to_idx = self.make_node_idx(nodes_numpy) # GNN 8.7%
        #node_to_idx = {node: idx for idx, node in enumerate(nodes_numpy)} 
        nodes_set = set(nodes_numpy)  # Convert to a set for faster lookup # GNN 2.1$
        edges_from = []
        edges_to = []
        
        for start_node in nodes_numpy: # GNN 4.7%
            start_idx = node_to_idx[start_node] # GNN 3.4%
            
            for dest_node in connect[start_node]: # GNN 34.7%
                if dest_node in nodes_set:        # GNN 20.6%
                    dest_idx = node_to_idx[dest_node] # GNN 18.3%
                    edges_from.append(start_idx) # GNN 3.6%
                    edges_to.append(dest_idx) # GNN 3.5%

        return [edges_from, edges_to]
    
    def make_localedge_numpy_agents(self, nodes_numpy, connect):
        return [self.make_localedge_numpy(nodes_numpy[i], connect) for i in range(len(nodes_numpy))]
    
    #@profile
    def make_node_idx(self, nodes_numpy):
        ret_dict = {}
        for idx, node in enumerate(nodes_numpy):
            if node not in ret_dict:
                ret_dict[node] = idx
        return ret_dict
    
    #@profile
    def find_indices(self, abst_list, nodes_numpy):
        indices = []
        nodes_to_idx = self.make_node_idx(nodes_numpy)
        indices = [nodes_to_idx[abst] for abst in abst_list]
        return indices
    
    #@profile
    def make_rows_agents(self, nodes_numpy, abst_arrays, connect):
        
        rows_common = np.array([np.array(self.find_indices(abst_arrays[i], nodes_numpy[i])) for i in range(self.env.num_agents)])
        
        
        nodes_numpy = np.reshape(nodes_numpy, (NUM_AGENTS, KNN_AGENTS, -1))
        abst_arrays = np.reshape(abst_arrays, (NUM_AGENTS, KNN_AGENTS,-1))
        
        #rows_common = np.reshape(rows_common, (NUM_AGENTS, KNN_AGENTS, -1))
        rows_local = np.array([np.array([self.find_indices(abst_arrays[i][robot], nodes_numpy[i][robot]) for robot in range(KNN_AGENTS)]) for i in range(self.env.num_agents)])
        
        return rows_common, rows_local
    
    def make_each_info(self, pre_each_info, rows_local):
        each_info = np.array(
            [
                np.array(
                    [
                        pre_each_info[agents][knnrobot][rows_local[agents][knnrobot]] for knnrobot in range(KNN_AGENTS)
                    ]
                ) for agents in range(self.env.num_agents)
            ]
            )
        return each_info
    
    #@profile
    def main(self):
        num_agents = MODE_AGENT[self.mode_id]
        len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, actionlist, picker_node, is_pickup, route = map_maker()
        self.env.map_init_reset(num_agents, len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, actionlist, picker_node, is_pickup, route)
        policy = DeepQPolicy(self.env.net_state_dim, self.env.net_act_dim, self.env.graph)
        policy.reset_env(actionlist, connect_dict, connect_to_dict)
        policy.set_num_agents(num_agents)

        frames = []
        model_file = self.tmpdirnamesla + "target_model.pt"
        policy.load_network(model_file, model_file)
        policy.reset_env(actionlist, connect_dict, connect_to_dict)
        frame_dir = self.dirframe
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)
        
        done = False
        observation, info = self.env.reset()
        node_info = info["nodeinfo"]
        
        if FRAME_SAVE_ON and self.frame_flag:
            frames.append(self.env.render())
            cv2.imwrite(self.dirframe+"/"+str(self.env.num_steps)+'.png', self.env.render())
        rt = 0
        rewards = []
        while not done:
            sub_graph_node, abst_array, pre_each_info = self.get_nodes(observation, node_info)
            local_edge = self.make_localedge_numpy_agents(sub_graph_node, self.env.connect_dict)                # 28.0%
            rows_common, rows_local = self.make_rows_agents(sub_graph_node, abst_array, self.env.connect_dict)  # 16.4%
            each_info = self.make_each_info(pre_each_info, rows_local)                                          #  1.8%
            
            action = policy.greedy(observation, (local_edge, rows_common, each_info))                           # 22.2%
            #action = policy.epsilon_greedy(observation, self.env.connect_dict)
            #print(self.env.state_nodes)
            #print(action)
            observation_next, reward, terminated, truncated, info_next = self.env.step(action)                          # 30.5%
            node_info_next = info_next["nodeinfo"]
            done = terminated or truncated
            rt += (GAMMA**self.env.num_steps)*reward
            rewards.append(reward)
            observation[:] = observation_next[:]
            node_info[:] = node_info_next[:]
            if FRAME_SAVE_ON and self.frame_flag:
                frames.append(self.env.render())
                cv2.imwrite(self.dirframe+"/"+str(self.env.num_steps)+'.png', self.env.render())        

        if FRAME_SAVE_ON and self.frame_flag:
            cv2.imwrite(self.dirnamesla+"/"+"last"+'.png', self.env.render())
        self.nooplist.append(self.env.cnt_noop)
        #self.pcolllist.append(self.env.cnt_privent_collision)
        self.pcolllist.append(self.env.cnt_all_privent_collision)
        self.append_task()
        #print(rt)

        
        #logging.info(f"Episode took {self.env.num_steps} timestep, return {rt}")
        
        if self.frame_flag:
            fig_rewards = plt.figure()
            plt.plot(rewards,label="rewards")
            plt.legend(loc=0)
            fig_rewards.savefig(self.dirnamesla+"rewards.png")
            plt.close()
        if FRAME_SAVE_ON and self.frame_flag:
            display_frames_as_gif(frames)
        self.frame_flag = False

    def loop_main(self):
        for i in tqdm(range(CHECK_LOOP_NUM)):
            self.main()
        self.frame_flag = True
        if SAVE_FLAG:
            self.save_list_txt(self.nooplist, self.diraveragesla + "noop.txt")
            self.save_list_txt(self.pcolllist, self.diraveragesla + "pcoll.txt")
            inoop_list = [noop+pcoll for noop, pcoll in zip(self.nooplist, self.pcolllist)]
            self.save_list_txt(inoop_list, self.diraveragesla + "ino_op.txt")
            self.save_list_txt(self.shorttasklist, self.diraveragesla + "short_task.txt")
            self.save_list_txt(self.longtasklist, self.diraveragesla + "long_task.txt")
            comptask_list = [short+long for short, long in zip(self.shorttasklist, self.longtasklist)]
            self.save_list_txt(comptask_list, self.diraveragesla+ "comptask.txt")

if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    #args = sys.argv
    #test = Testing()
    #test.tentime_main(args=args)
    #test.main(args)
    test = Testing(MODE_ID)
    test.loop_main()