import sys
import os
import pickle
import logging
import cv2


import numpy as np
import matplotlib.pyplot as plt
from MultiRobotsEnvironment import MultiRobotsEnvironment
#from DQL import DeepQPolicy
from CLASSIC import ClassicPolicy
from annealing import AnnealingPolicy
#from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from tqdm import tqdm

from CONST import  EXE_TIMESTEP, GAMMA, COMMONDIRSLA, DIRNAME, DIR_DICT, MODE_AGENT, TMPDIRNAME, RESULT_MOVE_FILE, DIRFRAME, NUM_AGENTS, MODE_ID
from map_maker import map_maker
FRAME_SAVE_ON = True
SAVE_FLAG = True
if SAVE_FLAG:
    CHECK_LOOP_NUM = 100
else:
    CHECK_LOOP_NUM = 1

def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    plt.axis('off')
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

    def main(self):
        num_agents = MODE_AGENT[self.mode_id]
        len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, actionlist, picker_node, is_pickup, route = map_maker()
        self.env.map_init_reset(num_agents, len_node, node_distances, connect_dict, connect_to_dict, coord_list, weights, actionlist, picker_node, is_pickup, route)
        #policy = ClassicPolicy(self.env)
        policy = AnnealingPolicy(self.env)
        policy.set_num_agents(num_agents)
        #policy.reset_env(actionlist, connect_dict, connect_to_dict)
        
        frames = []
        #policy = ClassicPolicy(self.env)
        frame_dir = self.dirframe
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)
        
        done = False
        observation = self.env.reset()
        if FRAME_SAVE_ON and self.frame_flag:
            frames.append(self.env.render())
            cv2.imwrite(self.dirframe+"/"+str(self.env.num_steps)+'.png', self.env.render())
        rt = 0
        rewards = []
        while not done:
            action = policy.greedy(observation)
            
            observation_next, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            rt += (GAMMA**self.env.num_steps)*reward
            rewards.append(reward)

            
            if FRAME_SAVE_ON and self.frame_flag:
                frames.append(self.env.render())
                cv2.imwrite(self.dirframe+"/"+str(self.env.num_steps)+'.png', self.env.render())        
        if FRAME_SAVE_ON and self.frame_flag:
            cv2.imwrite(self.dirnamesla+"/"+"last"+'.png', self.env.render())
        self.nooplist.append(self.env.cnt_noop)
        self.pcolllist.append(self.env.cnt_privent_collision)
        self.append_task()

        
        logging.info(f"Episode took {self.env.num_steps} timestep, return {rt}")
        if self.frame_flag:
            fig_rewards = plt.figure()
            plt.plot(rewards,label="rewards")
            plt.legend(loc=0)
            fig_rewards.savefig(self.dirnamesla+"rewards.png")
            plt.close()
        if FRAME_SAVE_ON and self.frame_flag:
            display_frames_as_gif(frames)
        self.frame_flag = False

    def tentime_main(self):
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
    test.tentime_main()