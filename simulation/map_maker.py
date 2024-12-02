from MAPDATA import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy as np
import random
from collections import deque
import networkx as nx
from TaskManager import *

#GRAPH
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 900
#IMAGE_WIDTH = 900
#IMAGE_HEIGHT = 600
MARGIN = 50
MARGIN_WIDTH = 70
MARGIN_HEIGHT = 50
ELEMENT_WIDTH = IMAGE_WIDTH - 2*MARGIN_WIDTH
ELEMENT_HEIGHT = IMAGE_HEIGHT - 2*MARGIN_HEIGHT


DIM_MAX_BFS = 7 

def create_directed_graph(rows=40, cols=40, removed_nodes=None, removes=0):
    if removed_nodes is None:
        removed_nodes = []

    graph = {}

    # ヘルパー関数: 特定のノードに隣接するノードのIDを計算
    def get_neighbor(node, direction):
        row, col = divmod(node, cols)
        if direction == 'up' and row > 0:
            return node - cols
        elif direction == 'down' and row < rows - 1:
            return node + cols
        elif direction == 'left' and col > 0:
            return node - 1
        elif direction == 'right' and col < cols - 1:
            return node + 1
        return None

    # ノードの配置とエッジの作成
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if node in removed_nodes:
                continue  # 削除されたノードは無視

            graph[node] = []
            
            if DIRECTON:
                #"""
                if row % 2 == 0:  # 偶数行
                    # 右へのエッジ
                    right_neighbor = get_neighbor(node, 'right')
                    if right_neighbor is not None and right_neighbor not in removed_nodes:
                        graph[node].append(right_neighbor)
                else:  # 奇数行
                    # 左へのエッジ
                    left_neighbor = get_neighbor(node, 'left')
                    if left_neighbor is not None and left_neighbor not in removed_nodes:
                        graph[node].append(left_neighbor)

                if col % 2 == 1:  # 奇数列
                    # 下へのエッジ
                    down_neighbor = get_neighbor(node, 'down')
                    if down_neighbor is not None and down_neighbor not in removed_nodes:
                        graph[node].append(down_neighbor)
                else:  # 偶数列
                    # 上へのエッジ
                    up_neighbor = get_neighbor(node, 'up')
                    if up_neighbor is not None and up_neighbor not in removed_nodes:
                        graph[node].append(up_neighbor)
                #"""
            else:
                #"""
                right_neighbor = get_neighbor(node, 'right')
                if right_neighbor is not None and right_neighbor not in removed_nodes:
                    graph[node].append(right_neighbor)
                
                left_neighbor = get_neighbor(node, 'left')
                if left_neighbor is not None and left_neighbor not in removed_nodes:
                    graph[node].append(left_neighbor)
                
                down_neighbor = get_neighbor(node, 'down')
                if down_neighbor is not None and down_neighbor not in removed_nodes:
                    graph[node].append(down_neighbor)

                up_neighbor = get_neighbor(node, 'up')
                if up_neighbor is not None and up_neighbor not in removed_nodes:
                    graph[node].append(up_neighbor)
                #"""
    # これより後ろは後ろは強連結成分分解をやることを検討
    x_graph = dict_to_networkx(graph)
    x_sccs = list(nx.strongly_connected_components(x_graph))
    scc = []
    len_scc = 0
    for x_scc in x_sccs:
        tmp_scc = list(x_scc)
        tmp_lenscc = len(tmp_scc)
        if len_scc < tmp_lenscc:
            scc = tmp_scc
            len_scc = tmp_lenscc
    new_graph = {}
    for startnode in scc:
        new_graph[startnode] = []
        for c_destnode in graph[startnode]:
            if c_destnode in scc:
                new_graph[startnode].append(c_destnode)
    graph = new_graph
    
    if len(graph)<MIN_LEN_NODE or len(graph)>MAX_LEN_NODE:
        return  create_directed_graph(rows=rows, cols=cols, removed_nodes=random.sample(range(rows*cols), removes), removes=removes)
    
    #graph = networkx_to_dict(remove_nodes_for_path_existence(dict_to_networkx(graph)))
    """
    #dead_ends = [node for node, edges in graph.items() if len(edges) == 0]
    nodes_with_incoming_edges = set()
    for destnodes in graph.values():
        for destnode in destnodes:
            nodes_with_incoming_edges.add(destnode)

    dead_ends = [node for node in graph if len(graph[node]) == 0 or (node not in nodes_with_incoming_edges)]

    if len(dead_ends) != 0:
        removed_nodes.extend(dead_ends)
        return create_directed_graph(rows, cols, removed_nodes=removed_nodes)

    while not (is_connected(graph=graph) and is_equal_or_larger_than_k_nodes(graph=graph, k=DIM_MAX_BFS)):
        graph = retain_largest_component(graph)
        graph = remove_nodes_with_fewer_than_k_nodes(graph, k=DIM_MAX_BFS)
    
    graph = retain_largest_component(graph)
    #"""
    return graph
    #assert is_connected(graph)
    #return graph

def dict_to_networkx(adj_list):
    G = nx.DiGraph()
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G

def networkx_to_dict(G:nx.DiGraph):
    adj_list = {node: list(G.neighbors(node)) for node in G.nodes()}
    return adj_list

def bfs(graph, start, k):
    """指定されたノードからのBFSを実行し、探索可能なノードの数を返す"""
    visited = set([start])
    queue = deque([start])
    while queue:
        if len(visited)>=k:
            break
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited)


def change_to_conncet_dict(connect_dict, label_node):
    ret_dict = {}
    for startnode, destnodes in connect_dict.items():
        ret_dict[label_node[startnode]] = []
        for dest in destnodes:
            ret_dict[label_node[startnode]].append(label_node[dest])
    return ret_dict

def change_to_position(position, label_node):
    ret_dict = {}
    for node, pos in position.items():
        ret_dict[label_node[node]] = pos
    return ret_dict

def shortest_path_matrix(direct):
    data = []
    row = []
    col = []
    for startnode, destnodes in direct.items():
        for destnode in destnodes:
            row.append(startnode)
            col.append(destnode)
            data.append(1)

    # 最大ノード番号を取得して行列のサイズを決定
    max_node = max(max(row, default=0), max(col, default=0))
    n = max_node + 1  # ノードは0から始まるため、最大ノード番号に1を加える
    
    csr = csr_matrix((data, (row, col)), shape=(n, n)).toarray()
    
    my_rount_int = lambda x: np.round((x*2+1)//2)
    return my_rount_int(shortest_path(csr)).astype(int)
    #return torch.tensor(my_rount_int(shortest_path(csr)).astype(int)).to(DEVICE)

def make_action_list(direct):
    action_dim = max(len(num_connect)+1 for num_connect in direct.values())
    actionlist = [[i for _ in range(action_dim)] for i in direct.keys()]
    for startnode, destnodes in direct.items():
        i = 1
        for destnode in destnodes:
            actionlist[startnode][i] = destnode
            i+=1
    #return torch.tensor(np.array(actionlist)).to(device=DEVICE)
    return actionlist

def make_coord_list(position):
    coord_list = []
    xmax = 0
    ymax = 0
    for k, v in position.items():
        tmpx, tmpy = v
        xmax = max(xmax, tmpx+1)
        ymax = max(ymax, tmpy+1)
    for k, v in position.items():
        coord_list.append( (int(ELEMENT_WIDTH*v[0]/xmax) + MARGIN_WIDTH, int(ELEMENT_HEIGHT*v[1]/ymax)+MARGIN_HEIGHT))
    return coord_list

def make_reverse_graph(direct):
    reverse_graph = {}
    for node, edges in direct.items():
        for edge in edges:
            if edge in reverse_graph:
                reverse_graph[edge].append(node)
            else:
                reverse_graph[edge] = [node]
    return reverse_graph

NUMTASK = 500

def make_factory(testmode):
    if testmode:
        rows = 20
        cols = 20
        removes = 80
        map_num_agent = 100
    else:
        if DIRECTON:
            rows = 16
            cols = 16
            removes = 60
            map_num_agent = 20
        else:
            rows = 10
            cols = 10
            removes = random.randrange(15,20)
            map_num_agent = 20
            
    FANTOM_CONNECTDICT = create_directed_graph(rows=rows, cols=cols, removed_nodes=random.sample(range(rows*cols), removes), removes=removes)
    FANTOM_POSITIONDICT = {node: (node // rows, node % cols) for node in FANTOM_CONNECTDICT}
    routenodes = np.array(list(FANTOM_CONNECTDICT.keys()))
    label_nodes = {routenodes[i]:i  for i in range(len(routenodes))}
    CONNECTDICT = change_to_conncet_dict(FANTOM_CONNECTDICT, label_nodes)
    POSITIONDICT = change_to_position(FANTOM_POSITIONDICT, label_nodes)

    PHYSICAL_CONNECT_DICT = CONNECTDICT
    LEN_NODE = len(CONNECTDICT)
    POST_PICKER_NODE = 0
    IS_PICKUP = [[random.randint(0,1)%2==0 for i in range(NUMTASK)] for _ in range(map_num_agent)]
    if LEN_NODE <= map_num_agent:
        return make_factory(testmode)
    startnode = np.random.choice(np.arange(LEN_NODE),map_num_agent,replace=False)
    route = list(np.random.randint(0, LEN_NODE, (map_num_agent, NUMTASK)))
    #print(len(route))
    #print(len(route[0]))
    for i in range(map_num_agent):
        route[i][0]=startnode[i]
    ROUTE = route
    return LEN_NODE, POST_PICKER_NODE, CONNECTDICT, PHYSICAL_CONNECT_DICT, POSITIONDICT, IS_PICKUP, ROUTE

COLORLIST = [
    (66, 245, 245), (203, 245, 66), (66, 245, 66), (66, 66, 245),
    (128, 0, 128), (0, 128, 128), (128, 128, 0), (128, 128, 128),
    (0, 245, 0), (245, 0, 245), (0, 0, 245), (66, 128, 245),
    ]

TEST_MODE = True
RANDOMMAP_MODE = True
DIRECTON = True # change
PICKERON = False
MIDDLE_PICKERON = False

MIN_LEN_NODE = 150
MAX_LEN_NODE = 200

TASK_ASSIGN_MODE = True
SINGLE_TASK_MODE = False

if TEST_MODE:
    IS_RANDOM = False
    #IS_RANDOM = True
    #MODE = HEAVYSQUARE_MODE # katayori
        
    #MODE = SUPERHEAVYSQUARE_MODE # katayori
    #MODE = NEWSQUAREPLUS_MODE # katayori
    #MODE = UAV4CUBET4_MODE
    MODE = FACTORY_LIKE
else:
    IS_RANDOM = True
    #MODE = HEAVYSQUARE_MODE
    # 学習環境を変えてみた。
    #MODE = NEWSQUAREPLUS_MODE # katayori
    ##MODE = SUPERHEAVYSQUARE_MODE # katayori
    #MODE = UAV4CUBET4_MODE
    
    MODE = FACTORY_LIKE
    

def map_maker():
    IS_PICKUP_FALSE = [False for _ in range(NUMTASK)]
    #MODE

    #MODE = WASIGNTOM_MODE
    #MODE = NEWSQUARE_MODE
    #MODE = NEWSQUAREPLUS_MODE # katayori
    #MODE = NEWSQUAREPLUS2_MODE # normal
    #MODE = SQUARE_MODE
    #MODE = HEAVYSQUARE_MODE # katayori
    

    
    if MODE == WASIGNTOM_MODE:
        LEN_NODE = WASIGNTOM_LENNODE
        POST_PICKER_NODE = 0
        if DIRECTON:
            CONNECTDICT = WASIGNTOM_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = WASIGNTOM_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = WASIGNTOM_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = WASIGNTOM_GRAPH_POSITION_DICT
        IS_PICKUP = WASIGNTOM_ROBOT_PICKUP
        ROUTE = WASIGNTOM_ROBOT_ROUTE
    elif MODE == NEWSQUARE_MODE:
        LEN_NODE = NEWSQUARE_NODE
        POST_PICKER_NODE = 0
        CONNECTDICT = NEWSQUARE_GRAPH_CONNECT_DIRECTON
        POSITIONDICT = NEWSQUARE_GRAPH_POSITION_DICT
        IS_PICKUP = NEWSQUARE_ROBOT_PICKUP
        ROUTE = NEWSQUARE_ROBOT_ROUTE
    elif MODE == NEWSQUAREPLUS_MODE:
        LEN_NODE = NEWSQUAREPLUS_NODE
        POST_PICKER_NODE = 96
        if DIRECTON:
            CONNECTDICT = NEWSQUAREPLUS_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = NEWSQUAREPLUS_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = NEWSQUAREPLUS_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = NEWSQUAREPLUS_GRAPH_POSITION_DICT
        IS_PICKUP = NEWSQUAREPLUS_ROBOT_PICKUP
        ROUTE = NEWSQUAREPLUS_ROBOT_ROUTE
    elif MODE == NEWSQUAREPLUS2_MODE:
        LEN_NODE = NEWSQUAREPLUS_NODE
        POST_PICKER_NODE = 0
        if DIRECTON:
            CONNECTDICT = NEWSQUAREPLUS_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = NEWSQUAREPLUS_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = NEWSQUAREPLUS_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = NEWSQUAREPLUS_GRAPH_POSITION_DICT
        IS_PICKUP = NEWSQUAREPLUS_ROBOT_PICKUP
        ROUTE = NEWSQUAREPLUS_ROBOT_ROUTE2
    elif MODE == SQUARE_MODE:
        LEN_NODE = SQUARE_NODE
        POST_PICKER_NODE = 0
        if DIRECTON:
            CONNECTDICT = SQUARE_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = SQUARE_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = SQUARE_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = SQUARE_GRAPH_POSITION_DICT
        IS_PICKUP = SQUARE_ROBOT_PICKUP
        ROUTE = SQUARE_ROBOT_ROUTE
    elif MODE == BIGRECT_MODE:
        LEN_NODE = BIGRECT_NODE
        POST_PICKER_NODE = 0
        if DIRECTON:
            CONNECTDICT = BIGRECT_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = BIGRECT_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = BIGRECT_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = BIGRECT_GRAPH_POSITION_DICT
        IS_PICKUP = BIGRECT_ROBOT_PICKUP
        ROUTE = BIGRECT_ROBOT_ROUTE
    elif MODE == HEAVYSQUARE_MODE:
        LEN_NODE = HEAVYSQUARE_NODE
        POST_PICKER_NODE = 0
        if DIRECTON:
            CONNECTDICT = HEAVYSQUARE_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = HEAVYSQUARE_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = HEAVYSQUARE_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = HEAVYSQUARE_GRAPH_POSITION_DICT
        IS_PICKUP = HEAVYSQUARE_ROBOT_PICKUP
        ROUTE = HEAVYSQUARE_ROBOT_ROUTE
    elif MODE == SUPERHEAVYSQUARE_MODE:
        LEN_NODE = SUPERHEAVYSQUARE_NODE
        POST_PICKER_NODE = 1160
        if DIRECTON:
            CONNECTDICT = SUPERHEAVYSQUARE_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = SUPERHEAVYSQUARE_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = SUPERHEAVYSQUARE_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = SUPERHEAVYSQUARE_GRAPH_POSITION_DICT
        IS_PICKUP = SUPERHEAVYSQUARE_ROBOT_PICKUP
        ROUTE = SUPERHEAVYSQUARE_ROBOT_ROUTE
    elif MODE == UAV4CUBE_MODE:
        LEN_NODE = UAV4CUBE_NODE
        POST_PICKER_NODE = 0
        if DIRECTON:
            CONNECTDICT = UAV4CUBE_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = UAV4CUBE_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = UAV4CUBE_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = UAV4CUBE_GRAPH_POSITION_DICT
        IS_PICKUP = UAV4CUBE_ROBOT_PICKUP
        ROUTE = UAV4CUBE_ROBOT_ROUTE
    elif MODE == UAV4CUBET4_MODE:
        LEN_NODE = UAV4CUBET4_NODE
        POST_PICKER_NODE = 0
        if DIRECTON:
            CONNECTDICT = UAV4CUBET4_GRAPH_CONNECT_DIRECTON
        else:
            CONNECTDICT = UAV4CUBET4_GRAPH_CONNECT_DIRECTOFF
        PHYSICAL_CONNECT_DICT = UAV4CUBET4_GRAPH_CONNECT_DIRECTOFF
        POSITIONDICT = UAV4CUBET4_GRAPH_POSITION_DICT
        IS_PICKUP = UAV4CUBET4_ROBOT_PICKUP
        ROUTE = UAV4CUBET4_ROBOT_ROUTE
    elif MODE == FACTORY_LIKE:
        if TEST_MODE == False:
            LEN_NODE = FACTORY_LEN_NODE
            POST_PICKER_NODE = 0
            CONNECTDICT = FACTORY_CONNECTDICT
            PHYSICAL_CONNECT_DICT = FACTORY_PHYSYCAL_CONNECT_DICT
            POSITIONDICT = FACTORY_POSITIONDICT
            IS_PICKUP = FACTORY_IS_PICKUP
            ROUTE = FACTORY_ROUTE
            LEN_NODE, POST_PICKER_NODE, CONNECTDICT, PHYSICAL_CONNECT_DICT, POSITIONDICT, IS_PICKUP, ROUTE = make_factory(testmode=TEST_MODE)
        else:
            LEN_NODE = FACTORY_LEN_NODE
            POST_PICKER_NODE = 0
            CONNECTDICT = FACTORY_CONNECTDICT
            PHYSICAL_CONNECT_DICT = FACTORY_PHYSYCAL_CONNECT_DICT
            POSITIONDICT = FACTORY_POSITIONDICT
            IS_PICKUP = FACTORY_IS_PICKUP
            ROUTE = FACTORY_ROUTE
        #print(IS_PICKUP)
        #print(ROUTE)
        
        #else:
            

    if PICKERON or MIDDLE_PICKERON:
        PICKER_NODE = POST_PICKER_NODE
    else:
        PICKER_NODE = 0

    #NODE_DISTANCES = shortest_path_matrix(CONNECTDICT)
    #NODE_PHYSICAL_DISTANCES = shortest_path_matrix(PHYSICAL_CONNECT_DICT)
    NODE_DISTANCES = shortest_path_matrix(CONNECTDICT)
    NODE_PHYSICAL_DISTANCES = shortest_path_matrix(PHYSICAL_CONNECT_DICT)
    ACTION_LIST = make_action_list(CONNECTDICT)
    COORD_LIST = make_coord_list(POSITIONDICT)
    CONNECT_TO_DICT = make_reverse_graph(CONNECTDICT)

    #NODES
    WEIGHTSZERO = [0 for _ in range(LEN_NODE)]
    WEIGHTSONES = [1 for _ in range(LEN_NODE)]
    WEIGHTSTWELVE = [12 for _ in range(LEN_NODE)]
    WEIGHTSFOUR = [4 for _ in range(LEN_NODE)]
    WEIGHTSSPECIAL = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    WEIGHTSDEFAULT = [2,1,1,2,3,2,1,1]
    #WEIGHTS = WEIGHTSSPECIAL
    #WEIGHTS = WEIGHTSTWELVE
    # for demo Weights = 4 but training 12
    WEIGHTS = WEIGHTSFOUR
    MAXWEIGHT = max(WEIGHTS)

    if TASK_ASSIGN_MODE: 
        if SINGLE_TASK_MODE:
            ROUTE = task_assignment.make_task(LEN_NODE)
        else:
            TASK_MANAGER = TaskManager(LEN_NODE, 15) #本当は「15」ではなく、「NUM_AGENTS」にしたい
            TASK_MANAGER.reset_tasks()
            return (LEN_NODE, NODE_DISTANCES, CONNECTDICT, CONNECT_TO_DICT, COORD_LIST, WEIGHTS, ACTION_LIST, PICKER_NODE, IS_PICKUP, TASK_MANAGER)
    return (LEN_NODE, NODE_DISTANCES, CONNECTDICT, CONNECT_TO_DICT, COORD_LIST, WEIGHTS, ACTION_LIST, PICKER_NODE, IS_PICKUP, ROUTE)
    #return (len_node, node_distances, connect_dict, connet_to_dict, coord_list, weights, actionlist, picker_node)




