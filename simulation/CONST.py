from map_maker import IS_RANDOM, TEST_MODE, DIRECTON
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
#DEVICE = torch.device("cpu")
##########################################
# MODE
RANDOMMODE = False
TASKADDMODE = False
MODE = "DQL"

DOUBLEMODE = True
DUELINGMODE = True
NOISYMODE = True

EACH_ROBOT_ACTION_MODE = True

SINGLE_AGENT_MODE = True
##########################################
# TimeStep Settings
if TEST_MODE:
    MAX_TIMESTEP = 1000
else:
    MAX_TIMESTEP = 150
    #MAX_TIMESTEP = 1000
MAX_TIMESTEP_TARGET = MAX_TIMESTEP
EXE_TIMESTEP = MAX_TIMESTEP_TARGET

##########################################
# DQL Epsode Settings
#MAX_TRAIN_EPS   = 1000 * 2
MAX_TRAIN_EPS   = 1000
#MAX_TRAIN_EPS   = 500 #* 5 #* 2
#MAX_TRAIN_EPS   = 700
#MAX_TRAIN_EPS   = 500
#TRAIN_EPS   = 100
TRAIN_EPS   = MAX_TRAIN_EPS
#TRAIN_EPS   = 1

LOOPS = (MAX_TRAIN_EPS + TRAIN_EPS - 1) // TRAIN_EPS # 切り上げ
##########################################
# ROBOT,GRAPH Settings

NUM_TASK = MAX_TIMESTEP

DIR_DICT            = ["Z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J","K","L"]
DIR_DICT[10] = "A"
MODE_AGENT2         = [0,2,4,6,8]
MODE_AGENT3         = [0,3,6,9,12]
MODE_AGENT_PLUS2    = [0,6,7,8,9]
MODE_AGENT_NORMAL   = [0,2,3,4,5,6,7,8,9]
#MODE_AGENT_NORMAL_4   = [0,5,6,7,8,9,10,11,12]
MODE_AGENT_TRAIN_EX   = [0,12,2,3,4,5,6,7,8,9,10,11,12]
MODE_AGENT_TRAIN_MIN_EX   = [0,6,2,3,4,5,6,7,8,9,10,11,12]
MODE_AGENT_TEST_EX   = [0,15,2,3,4,5,6,7,8,9,10,11,12]

#MODE_AGENT_TRAIN_EX   = [0,6,2,3,4,5,6,7,8,9,10,11,12]
#MODE_AGENT_TEST_EX   = [0,4,2,3,4,5,6,7,8,9,10,11,12]

#MODE_AGENT = MODE_AGENT_NORMAL
NUM_KNN = 6
NUM_REWARD_KNN = 5
DISTANCE = 7

if TEST_MODE:
    MODE_AGENT = MODE_AGENT_TEST_EX
else:
    MODE_AGENT = MODE_AGENT_TRAIN_EX
    MODE_MIN_AGENT = MODE_AGENT_TRAIN_MIN_EX

MODE_ID = 10
#MODE_ID = 1
NUM_AGENTS  = MODE_AGENT[MODE_ID]
KNN_AGENTS = min(NUM_KNN, NUM_AGENTS)
KNN_REWARD_AGENTS = min(NUM_REWARD_KNN, NUM_AGENTS)

#'''
#
if IS_RANDOM:
    TASKRANDOMMODE = True
else:
    TASKRANDOMMODE = False
TASKRANDOMSTART = 0# MAX_TRAIN_EPS//4
TASKRANDOMFREQ = 1

#'''
##########################################
# Embedding Settings

FULLNODEDATAMODE = False # Encoder
NODEDATAMODE = False
PARTIALMODE = False
#PARTIAL_AGENT_MODE = False
ALLPARTIALMODE = False
ALLFULLNODEDATAMODE = False #Incomplete implement so it is always false
SINGLESIGHTMODE = False
SINGLEGCNMODE = False

PARTIAL_AGENT_MODE = False

LOCALGCNMODE = True

AGENT_RANDOM_MODE = False
DIST_MODE = False

# GNNの場合バッチのグラフが変わるためedgeを利用するのが困難になった
# randommapでは袋小路　1->2=3 のような形を回避できないために双方向のグラフを用意する必要が発生

if DIRECTON:
    dim_action = 2
else:
    dim_action = 4

NUM_NODES_OF_LOCAL_GRAPH_FROM_START = 1 + dim_action #+ dim_action*dim_action #+ 0
NUM_NODES_OF_LOCAL_GRAPH_TO_START = 0 #+ dim_action
NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST = 1 #+ dim_action
NUM_NODES_OF_LOCAL_GRAPH_TO_DEST = 0 #+ dim_action

NUM_NODES_OF_LOCAL_GRAPH_START = NUM_NODES_OF_LOCAL_GRAPH_FROM_START + NUM_NODES_OF_LOCAL_GRAPH_TO_START
NUM_NODES_OF_LOCAL_GRAPH_DEST = NUM_NODES_OF_LOCAL_GRAPH_FROM_DEST + NUM_NODES_OF_LOCAL_GRAPH_TO_DEST

KNN_MODE = True

#NUMNEXTSTEP = 3
NUMNEXTSTEP = 1 
WITHTARGET = True

FULLNODE_MAXVIEW_RANGE = 1

##########################################
# Env Settings
PENALTY_NO_EXCEPTION = 1.0
PENALTY_COLLISION = 0.0 # before 0.5 now 10
# PENALTY_COLLISION = 2/NUM_AGENTS
#PENALTY_PRIVENT_COLLISION= 0.01
#PENALTY_PRIVENT_COLLISION= 0.1
PENALTY_PRIVENT_COLLISION= 0.1
PENALTY_NO_OP = 0.0
PENALTY_PRIORITY = 0.0
##########################################
#DQL Settings
GAMMA = 0.99
SYNC_FREQ   = 500
TRAIN_FREQ  = 5
EPSILON     = 1.0
MINEPSILON  = 0.10
LINEAREPSMODE = False
if NOISYMODE:
    EPSILON = 1.0#最初の1epsだけランダム
    MINEPSILON = 0.0

NEXTEPSILON_PARAMETER = EPSILON/MAX_TRAIN_EPS
NEXTEPSILONS_GAMMA = MINEPSILON ** (1/MAX_TRAIN_EPS)
BUF_CAPASITY = 2048*32#*10

#LEARNINGRATE = 0.001
#LEARNINGRATE = 0.001 #AC
LEARNINGRATE = 1e-4
#if LOCALGCNMODE:
#    LEARNINGRATE = 1e-5

#LEARNINGRATE = 0.00005 # 変えてみたぞ杉本20240130
#LEARNINGRATE = 1e-6 # 変えてみたぞ杉本20240130
#BUF_CAPASITY = 1
#BUF_CAPASITY = 2048
#BUF_CAPASITY = MAX_TRAIN_EPS*MAX_TIMESTEP

#BATCH_SIZE  = 64
#BATCH_SIZE  = 64 #* 2 # 変えてみたぞ杉本20240201
#BATCH_SIZE  = 1
BATCH_SIZE  = 64
INI_BEST_RETURN = -100000
##########################################
# filename
COMMONDIR = "./dataR"
COMMONDIRSLA = COMMONDIR + "/"
DIRNAME = "dataR"
TMPDIRNAME = "tmpdata"
DIRFRAME = "frame"

EMBEDDINGS_FILE = COMMONDIRSLA + "embedding.pkl"
TRAIN_MODEL_FILE = "train_model.pt"
TARGET_MODEL_FILE = "target_model.pt"
NPY_RETURN = "returns"
NPY_RETURN_FILE = NPY_RETURN + ".npy"
NPY_TARGET_RETURN = "targetreturns"
NPY_TARGET_RETURN_FILE = NPY_TARGET_RETURN + ".npy"
LOSS_FILE = "losses"

RESULT_MOVE_FILE = "move.txt"
MODEL_DESCRIBE_FILE = "model.txt"

#########################################









