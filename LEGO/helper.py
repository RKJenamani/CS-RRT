import random
import numpy as np

import math

def state_to_numpy(state):
    strlist = state.split()
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

def numpy_to_state(array):
    state = ""
    for i in range(len(array)):
        state += str(array[i])+" "
    return state

def is_point_free1(conf, occ_g, inc = 0.04):
    px, py = conf[0], conf[1]
    lx, ly = px-inc, py
    rx, ry = px+inc, py
    ux, uy = px, py+inc
    dx, dy = px, py-inc
    pc = []
    dim = occ_g.shape[0]
    pc = [(px, py), (ux, uy), (dx, dy),(lx, ly), (rx, ry)]
    for p in pc:
        x, y = int(dim*p[0]), int(dim*p[1])
        x = max(min(x, dim-1), 0)
        y = max(min(y, dim-1), 0)
        if(not occ_g[x][y]):
            return 0
    return 1

def is_edge_free1(node1_pos, node2_pos, occ_g, EDGE_DISCRETIZATION = 100, inc = 0.04):
    diff = node2_pos - node1_pos
    step = diff/EDGE_DISCRETIZATION   
    for i in range(EDGE_DISCRETIZATION+1):
        nodepos = node1_pos + step*i
        if(not is_point_free1(nodepos, occ_g, inc)):
            return 0
    return 1

def get_valid_start_goal(dense_G, occ_g1, occ_g2, inc):
    start_n = random.choice(list(dense_G.nodes()))
    goal_n = random.choice(list(dense_G.nodes()))

    start = state_to_numpy(dense_G.node[start_n]['state'])
    goal = state_to_numpy(dense_G.node[goal_n]['state'])

    while is_edge_free(start, goal, occ_g1, inc = inc) or not (is_point_free(start, occ_g2, inc) and is_point_free(goal, occ_g2,inc)):
        start_n = random.choice(list(dense_G.nodes()))
        goal_n = random.choice(list(dense_G.nodes()))

        start = state_to_numpy(dense_G.node[start_n]['state'])
        goal = state_to_numpy(dense_G.node[goal_n]['state'])

    return start_n, goal_n

def calc_weight_states(s1, s2):
    config1 = state_to_numpy(s1)
    config2 = state_to_numpy(s2)
    return np.linalg.norm(config1-config2)

def is_point_free(conf, occ_g, inc=0):
    inc = 0
    dim = occ_g.shape[0]
    px, py = conf[0], conf[1]
    x = max(min(int(dim*px),dim-1), 0)
    y = max(min(int(dim*py),dim-1), 0)
    if not occ_g[x][y]:
        return 0

    if inc != 0:
        pc = []
        angular_resolution = 0.314*2
        number_of_points = int(2*3.14/angular_resolution)
        for i in range(number_of_points+1):
            y = py + inc*math.sin(i*angular_resolution)
            x = px + inc*math.cos(i*angular_resolution)
            pc.append((x,y))
            
        for p in pc:
            if p[0]<0 or p[0]>1 or p[1]<0 or p[1]>1 :
                return 0
            x, y = int(dim*p[0]), int(dim*p[1])
            if(not occ_g[x][y]):
                return 0
    return 1

def get_valid_start_goal(dense_G, occ_g1, occ_g2, inc):
    start_n = random.choice(list(dense_G.nodes()))
    goal_n = random.choice(list(dense_G.nodes()))

    start = state_to_numpy(dense_G.node[start_n]['state'])
    goal = state_to_numpy(dense_G.node[goal_n]['state'])

    while is_edge_free(start, goal, occ_g1, inc = inc) or not (is_point_free(start, occ_g2, inc) and is_point_free(goal, occ_g2,inc)):
        start_n = random.choice(list(dense_G.nodes()))
        goal_n = random.choice(list(dense_G.nodes()))

        start = state_to_numpy(dense_G.node[start_n]['state'])
        goal = state_to_numpy(dense_G.node[goal_n]['state'])

    return start_n, goal_n

def is_edge_free(node1_pos, node2_pos, occ_g, inc=0):
    if np.allclose(node1_pos, node2_pos):
        if(not is_point_free(node1_pos, occ_g)):
            return 0
        return 1
    diff = node2_pos - node1_pos
    resolution = 0.001
    length = np.linalg.norm(diff)
    step = (resolution/length)*diff
    number_of_checks = int(length/resolution)
    for i in range(number_of_checks+1):
        nodepos = node1_pos + step*i
        if(not is_point_free(nodepos, occ_g)):
            return 0
    if(not is_point_free(node2_pos, occ_g)):
            return 0    
    return 1

def add_s_g_to_graph(G, start_s, goal_s):

    wt_s = []
    wt_g = []

    for k in list(G.nodes()):
        w_s = calc_weight_states(start_s, G.nodes[k]['state'])
        w_g = calc_weight_states(goal_s, G.nodes[k]['state'])
        wt_s.append([w_s, k])
        wt_g.append([w_g, k])

    wt_s.sort()
    wt_g.sort()

    G.add_node('s', state = start_s)
    G.add_node('g', state = goal_s)

    count = 0

    while count < 10:
        n = wt_s[count][1]
        G.add_edge('s', n)
        G['s'][n]['weight'] = wt_s[count][0]

        n = wt_g[count][1]
        G.add_edge('g', n)
        G['g'][n]['weight'] = wt_g[count][0]

        count += 1
