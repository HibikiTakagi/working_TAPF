from MultiRobotsEnvironment import MultiRobotsEnvironment
from CONST import (
    TRAIN_EPS,TRAIN_FREQ,BATCH_SIZE,SYNC_FREQ,GAMMA, COMMONDIRSLA,DIRNAME,DIR_DICT,TMPDIRNAME,
    MODEL_DESCRIBE_FILE, TRAIN_MODEL_FILE, TARGET_MODEL_FILE,NPY_TARGET_RETURN_FILE,
    NPY_RETURN_FILE,NPY_TARGET_RETURN,NPY_RETURN,LOSS_FILE, NUM_AGENTS,
    KNN_MODE,
    MODE_ID, MODE_AGENT,AGENT_RANDOM_MODE,MODE_MIN_AGENT,
    NUM_NODES_OF_LOCAL_GRAPH_START, NUM_NODES_OF_LOCAL_GRAPH_DEST,KNN_AGENTS
)
from map_maker import map_maker, FACTORY_LIKE, IS_RANDOM, RANDOMMAP_MODE, MODE
from DQL import DeepQPolicy
from replay_buffer import ReplayBuffer
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from graphDB import NODE_INFO_ROUTE, NODE_DATA_DIM, NODE_DATA_EACH_DIM
import cv2

class Training:
    """ This class provides training function on multi robot env(env)
    
    Attributes:
        mode_id():
        dirname(string):
        dirnamesla(string):
        tmpdirname(string):
        tmpdirnamesla(string):
        env(MultiRobotsEnvironment): environment(robots' position, map, etc...)
        replay_buffer(ReplayBuffer):
    """
    def __init__(self, mode_id) -> None:
        self.mode_id = mode_id
        self.dirname = COMMONDIRSLA + DIRNAME + DIR_DICT[self.mode_id]
        self.dirnamesla = self.dirname + "/"
        self.tmpdirname = self.dirnamesla + TMPDIRNAME
        self.tmpdirnamesla = self.tmpdirname + "/"
        self.set_directry()
        self.env = MultiRobotsEnvironment(self.mode_id)
        self.replay_buffer = ReplayBuffer()
        pass
        
    def set_directry(self):
        self.model_file = self.dirnamesla + MODEL_DESCRIBE_FILE

        self.tmptrain_file = self.tmpdirnamesla + TRAIN_MODEL_FILE
        self.tmptarget_file = self.tmpdirnamesla + TARGET_MODEL_FILE
        self.tmptargetreturns_file = self.tmpdirnamesla + NPY_TARGET_RETURN_FILE
        self.tmpreturns_file = self.tmpdirnamesla + NPY_RETURN_FILE
        self.tmptargetreturns = self.tmpdirnamesla + NPY_TARGET_RETURN
        self.tmpreturns = self.tmpdirnamesla + NPY_RETURN
        self.tmplosses = self.tmpdirnamesla + LOSS_FILE
        self.boarddir = self.tmpdirnamesla + 'log'

    def save_fig_conv_return(self, returns, target_returns):
        """ this function saves convergence graph
        
            Args:
                returns(?):trained data?
                target_returns(?):data?
        """
        fig_convergence_return, ax = plt.subplots()
        plt.plot(returns,label="train_return") 
        plt.plot(target_returns,label="target_return")
        ax.set_xlabel('episodes')
        ax.set_ylabel('Returns')
        ax.grid(True)
        plt.legend(loc=0)
        fig_convergence_return.savefig(self.dirnamesla+"convergence.png")
        plt.clf()
        plt.close()
    
    def save_fig_losses(self, losses):
        """ this function saves loss graph
        
            Args:
                losses(?):loss of the model
        """
        fig_loss, ax = plt.subplots()
        #plt.plot(losses, label='loss')
        x_eps, y_loss = zip(*losses)
        #plt.scatter(x_eps,y_loss, label='loss')
        plt.plot(x_eps,y_loss, '-',label='loss')
        ax.set_xlabel('episodes')
        ax.set_ylabel('loss')
        ax.grid(True)
        plt.legend(loc=0)
        fig_loss.savefig(self.dirnamesla+"loss.png")
        plt.clf()
        plt.close()
        
    def get_num_agents(self):
        return random.randint(MODE_MIN_AGENT[self.mode_id], MODE_AGENT[self.mode_id])
    
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
                            pre_each_info[agents][knn_robot][rows_local[agents][knn_robot]] for knn_robot in range(KNN_AGENTS)
                        ]
                    ) for agents in range(self.env.num_agents)
                ]
            )
        return each_info


    #@profile
    def main(self):
        num_agents = MODE_AGENT[self.mode_id]
        len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, action_list, picker_node, is_pickup, route = map_maker()
        self.env.map_init_reset(num_agents, len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, action_list, picker_node, is_pickup, route)
        #replay_buffer = ReplayBuffer()
        policy = DeepQPolicy(self.env.net_state_dim, self.env.net_act_dim, self.env.graph)
        policy.reset_env(action_list, connect_dict, connect_to_dict)
        policy.set_num_agents(self.env.num_agents)

        sync_cnt=0
        target_returns = [] # target return log
        returns = [] # return log
        losses = [] # loss log
        num_of_train = 0

        with open(self.model_file, "w") as f:
                f.write(f'{policy._q_network}')

        
        for eps in tqdm(range(TRAIN_EPS)): # repeat for DQN train episode
            if MODE==FACTORY_LIKE and RANDOMMAP_MODE:
                if AGENT_RANDOM_MODE :
                    num_agents = self.get_num_agents()
                    policy.set_num_agents(num_agents)
                len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, action_list, picker_node, is_pickup, route = map_maker()
                self.env.map_init_reset(num_agents, len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, action_list, picker_node, is_pickup, route)
                policy.reset_env(action_list, connect_dict, connect_to_dict)
                
                #print(self.env.len_node)
            elif AGENT_RANDOM_MODE:
                num_agents = self.get_num_agents()
                policy.set_num_agents(num_agents)
                len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, action_list, picker_node, is_pickup, route = map_maker()
                self.env.map_init_reset(num_agents, len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, action_list, picker_node, is_pickup, route)
            #print(self.env.len_node)
            # var init
            log_eps_return = 0
            done = False
            observation, info = self.env.reset()
            observation_next = observation.copy()
            replay_obs = self.env.replay_obs.copy()
            replay_obs_next = self.env.replay_obs_next.copy() # obs next間のエージェントが種類が違うせいで性能落ち？
            node_info = info["nodeinfo"]
            
            while not done: # Do task and train in one episode while not terminated or truncated
                #action = policy.epsilon_greedy(observation) # get next action from policy
                is_picking_now = self.env.robot_envs[0].pick_timer > 0
                
                sub_graph_node, abst_array, pre_each_info= self.get_nodes(observation, node_info)
                local_edge = self.make_localedge_numpy_agents(sub_graph_node, self.env.connect_dict)
                rows_common, rows_local = self.make_rows_agents(sub_graph_node, abst_array, self.env.connect_dict)
                each_info = self.make_each_info(pre_each_info, rows_local)
                
                action = policy.epsilon_greedy(observation, (local_edge, rows_common, each_info)) # get next action from policy # GNN 14.8%
                
                observation_next, reward, terminated, truncated, info_next = self.env.step(action) # do action
                node_info_next = info_next["nodeinfo"]
                
                sub_graph_node_next, abst_array_next, pre_each_info_next = self.get_nodes(observation_next, node_info_next)
                local_edge_next = self.make_localedge_numpy_agents(sub_graph_node_next, self.env.connect_dict)
                rows_common_next, rows_local_next = self.make_rows_agents(sub_graph_node_next, abst_array_next, self.env.connect_dict)
                each_info_next = self.make_each_info(pre_each_info_next, rows_local_next)

                done = terminated or truncated
                if KNN_MODE: # What is KNN mode?
                    replay_obs_next[:] = self.env.replay_obs_next[:]

                log_eps_return += (GAMMA**self.env.num_steps)*reward # reward decreases as step increases
                sync_cnt += 1
                # record replay
                
                if KNN_MODE:
                    #self.replay_buffer.add((replay_obs.copy(), action.copy(), reward, replay_obs_next.copy(), terminated))
                    if is_picking_now:
                        pass
                    else:
                        self.replay_buffer.add(
                            (replay_obs.copy(), action[0], reward, replay_obs_next.copy(), terminated, 
                            local_edge[0], rows_common[0], each_info[0],
                            local_edge_next[0], rows_common_next[0], each_info_next[0]))
                    #self.replay_buffer.add((replay_obs.copy(), action[0], reward, replay_obs_next.copy(), terminated, self.env.connect_dict))
                else:
                    self.replay_buffer.add((observation.copy(), action.copy(), reward, observation_next.copy(), terminated))
                observation[:] = observation_next[:] # get updated ovservation(env's situation?)
                node_info[:] = node_info_next[:]
                
                if KNN_MODE:
                    replay_obs[:] = self.env.replay_obs[:]
                
                #print(f"e:{eps},ts{self.env.num_steps},a:{action},r:{reward:.3f},o:{replay_obs[NODE_DATA_DIM*action+NODE_INFO_ROUTE]}")
                if len(self.replay_buffer) < BATCH_SIZE:
                    continue

                if self.env.num_steps % TRAIN_FREQ == 0:
                    policy.train_batch(batch=self.replay_buffer.sample(BATCH_SIZE)) # TRAIN from replay # GNN 73.0%
            
            else:
                returns.append(log_eps_return)
                if sync_cnt>=SYNC_FREQ:
                    policy.sync() # update(sync) policy from trained data when enough steps have ended
                    sync_cnt=0
            
            losses.append([eps, policy.loss])
            
            cv2.imwrite(self.dirnamesla+"/"+"factory"+"/"+str(eps)+"-"+str(log_eps_return)+'.png', self.env.render())
            #if policy.loss < 1e-2:
                #print(f"loss:{policy.loss}")
            #    print(f"eps:{eps}")
            #    break
            #print("egui")

            if sync_cnt==0 or eps==0: # if policy has synchronised or first episode
                log_target_return = 0
                done = False
                observation, info = self.env.reset()
                node_info = info["nodeinfo"]
                while not done:
                    #action = policy.target_greedy(observation)
                    #local_edge = self.make_localedge_numpy_agents(self.get_nodes(observation), self.env.connect_dict)
                    
                    sub_graph_node, abst_array, pre_each_info = self.get_nodes(observation, node_info)
                    local_edge = self.make_localedge_numpy_agents(sub_graph_node, self.env.connect_dict)
                    rows_common, rows_local = self.make_rows_agents(sub_graph_node, abst_array, self.env.connect_dict)
                    each_info = self.make_each_info(pre_each_info, rows_local)

                    action = policy.target_greedy(observation, (local_edge, rows_common, each_info)) # GNN4.9%
                    
                    observation_next, reward, terminated, truncated, info_next = self.env.step(action) 
                    node_info_next = info_next["nodeinfo"]
                    
                    done = terminated or truncated
                    log_target_return += (GAMMA**self.env.num_steps)*reward
                    observation[:] = observation_next[:]
                    node_info[:] = node_info_next[:]
            
            if num_of_train%100==0:
                # save return and losses log every 100 loops
                self.save_fig_conv_return(returns=returns, target_returns=target_returns)
                self.save_fig_losses(losses=losses)

                policy.save_network(self.tmptrain_file,self.tmptarget_file)
                paramtargetfile = self.tmptarget_file+"h"
                policy.save_network_param(self.tmptrain_file, paramtargetfile)

            target_returns.append(log_target_return)
            policy.setepsilon()
            num_of_train += 1

        np.save(self.tmpreturns, returns)
        np.save(self.tmptargetreturns, target_returns)
        np.save(self.tmplosses, losses)
        policy.save_network(self.tmptrain_file, self.tmptarget_file)

        self.save_fig_conv_return(returns=returns, target_returns=target_returns)
        self.save_fig_losses(losses=losses)

        writer = SummaryWriter(self.boarddir)
        for i in range(len(target_returns)):
            writer.add_scalar("Returns", returns[i], i)
            writer.add_scalar("Target Returns", target_returns[i], i)
            #writer.add_scalar("loss", losses[i], i)
        writer.close()
        
if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    #args = sys.argv
    #test = Testing()
    #test.tentime_main(args=args)
    #test.main(args)
    train = Training(MODE_ID)
    train.main()
