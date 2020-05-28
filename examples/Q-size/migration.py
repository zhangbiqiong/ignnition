# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2], Arnau Badia [^3]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu
#
# [^3]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: abadia@ac.upc.edu


from __future__ import print_function

import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import os
import tensorflow as tf
import re
import random
import tarfile
import json
import sys
import argparse


def genPath(R,s,d,connections):
    while s != d:
        yield s
        s = connections[s][R[s,d]]
    yield s

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def load_routing(routing_file):
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R=R.drop([R.shape[0]], axis=1)
    return R.values

def make_indices(paths):
    node_indices=[]
    path_indices=[]
    sequ_indices=[]
    segment=0
    for p in paths:
        node_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment +=1
    return node_indices, path_indices, sequ_indices

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def extract_links(n, connections, link_cap):
    A = np.zeros((n,n))

    for a,c in zip(A,connections):
        a[c]=1

    G=nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges=list(G.edges)
    capacities_links = []
    # The edges 0-2 or 2-0 can exist. They are duplicated (up and down) and they must have same capacity.
    for e in edges:
        if str(e[0])+':'+str(e[1]) in link_cap:
            capacity = link_cap[str(e[0])+':'+str(e[1])]
            capacities_links.append(capacity)
        elif str(e[1])+':'+str(e[0]) in link_cap:
            capacity = link_cap[str(e[1])+':'+str(e[0])]
            capacities_links.append(capacity)
        else:
            print("ERROR IN THE DATASET!")
            exit()
    return edges, capacities_links

def make_paths(R,connections, link_cap):
    n = R.shape[0]
    edges, capacities_links = extract_links(n, connections, link_cap)
    # link paths
    link_paths = []
    for i in range(n):
        for j in range(n):
            if i != j:
                link_paths.append([edges.index(tup) for tup in pairwise(genPath(R,i,j,connections))])


    node_paths = []
    for i in range(n):
        for j in range(n):
            if i != j:
                node_paths.append(list(genPath(R,i,j,connections))[:-1])

    return link_paths, node_paths, capacities_links

class Parser:
    netSize = 0
    offsetDelay = 0
    hasPacketGen = True

    def __init__(self,netSize):
        self.netSize = netSize
        self.offsetDelay = netSize*netSize*3

    def getBwPtr(self,src,dst):
        return ((src*self.netSize + dst)*3)
    def getGenPcktPtr(self,src,dst):
        return ((src*self.netSize + dst)*3 + 1)
    def getDropPcktPtr(self,src,dst):
        return ((src*self.netSize + dst)*3 + 2)
    def getDelayPtr(self,src,dst):
        return (self.offsetDelay + (src*self.netSize + dst)*8)
    def getJitterPtr(self,src,dst):
        return (self.offsetDelay + (src*self.netSize + dst)*8 + 7)

def ned2lists(f):
    channels = []
    link_cap = {}
    queue_sizes_dict = {}

    node_id_regex = re.compile(r'\s+node(\d+): Server {')
    queue_size_regex = re.compile(r'\s+queueSize = (\d+);')
    current_node_id = None
    for line in f:
        line = line.decode()
        m1 = node_id_regex.match(line)
        if m1:
            current_node_id = int(m1.groups()[0])
        m2 = queue_size_regex.match(line)
        if m2:
            queue_sizes_dict[current_node_id] = int(m2.groups()[0])

    queue_sizes = [queue_sizes_dict[key] for key in sorted(queue_sizes_dict.keys())]
    f.seek(0)

    p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps+\s<-->\s+node(\d+).port\[(\d+)\]')
    for line in f:
        line = line.decode()
        m=p.match(line)
        if m:
            auxList = []
            it = 0
            for elem in list(map(int,m.groups())):
                if it!=2:
                    auxList.append(elem)
                it = it + 1
            channels.append(auxList)
            link_cap[(m.groups()[0])+':'+str(m.groups()[3])] = int(m.groups()[2])

    n=max(map(max, channels))+1
    connections = [{} for _ in range(n)]
    # Shape of connections[node][port] = node connected to
    for c in channels:
        connections[c[0]][c[1]]=c[2]
        connections[c[2]][c[3]]=c[0]

    connections = [[v for k,v in sorted(con.items())]
                   for con in connections ]
    return connections, n, link_cap, queue_sizes


def get_corresponding_values(posParser, line, n, bws, delays, jitters):
    bws.fill(0)
    delays.fill(0)
    jitters.fill(0)
    it = 0
    for i in range(int(n)):
        for j in range(int(n)): #this n is incorrect. should be 14?
            if i!=j:
                delay = posParser.getDelayPtr(i, j)
                jitter = posParser.getJitterPtr(i, j)
                traffic = posParser.getBwPtr(i, j)
                bws[it] = float(line[traffic])
                delays[it] = float(line[delay])
                jitters[it] = float(line[jitter])
                it = it + 1


#######################
def make_jsons(ned_file, routing_file, data_file):
    con,n,link_cap, queue_sizes = ned2lists(ned_file)
    posParser = Parser(n)

    R = load_routing(routing_file)
    link_paths, node_paths, link_capacities = make_paths(R, con, link_cap)

    n_link_paths = len(link_paths)
    n_node_paths = len(node_paths)
    assert n_link_paths == n_node_paths
    n_paths = n_link_paths

    n_links = max(max(link_paths)) + 1
    n_nodes = max(max(node_paths)) + 1

    a = np.zeros(n_paths)
    d = np.zeros(n_paths)
    jit = np.zeros(n_paths)


    link_indices, link_path_indices, link_sequ_indices = make_indices(link_paths)
    node_indices, node_path_indices, node_sequ_indices = make_indices(node_paths)

    result_list = []
    for line in data_file:  #for each sample
        line = line.decode().split(',')
        get_corresponding_values(posParser, line, n, a, d, jit)


        data = {
            'traffic': a.tolist(),
            'delay': d.tolist(),
            'jitter': jit.tolist(),
            'link_capacity': link_capacities,
            'queue_sizes': queue_sizes,
        }



        # Here we find the definition of the graph and write them to topology.json
        dict_paths = {}
        dict_links = {}
        dict_node = {}  #matches the index and the name of the entity node

        # here we create the entities
        data['entities'] = {}
        for j in range(0, n_paths):
            name = "p" + str(j)
            data['entities'][name] = "path"
            dict_paths[str(j)] = name


        for j in range(0, n_links):
            name = "l" + str(j)
            data['entities'][name] = "link"
            dict_links[str(j)] = name

        for j in range(0, n_nodes):
            name = "n" + str(j) #n0,n1,n2,n3...
            data['entities'][name] = "node"
            dict_node[str(j)] = name    #4 -> n1 (e.g matches the index 0 to n0 )


        #here we create the adjcecencies
        data['adj_paths_links'] = {}
        data['adj_links_paths'] = {}
        data['adj_nodes_paths'] = {}
        data['adj_paths_nodes'] = {}

        for j in range(0, len(link_indices)):
            p = str(link_path_indices[j])
            l = str(link_indices[j])

            name_path = dict_paths[p]
            name_link = dict_links[l]

            if not name_link in data['adj_paths_links'].keys():
                data['adj_paths_links'][name_link] = []

            if not name_path in data['adj_links_paths'].keys():
                data['adj_links_paths'][name_path] = []

            data['adj_paths_links'][name_link].append(name_path)
            data['adj_links_paths'][name_path].append(name_link)


        for j in range(0, len(node_indices)):
            p = str(node_path_indices[j])   #path of each of the nodes respectively
            num = str(node_indices[j])    #each of the nodes

            name_path = dict_paths[p]
            name_node = dict_node[num]

            if not name_path in data['adj_nodes_paths'].keys():
                data['adj_nodes_paths'][name_path] = []

            if not name_node in data['adj_paths_nodes'].keys():
                data['adj_paths_nodes'][name_node] = []

            data['adj_nodes_paths'][name_path].append(name_node)
            data['adj_paths_nodes'][name_node].append(name_path)

        data['path_interleave'] = ['node','link']

        result_list.append(data)

    with open('data.txt', 'a') as outfile:
        json.dump(result_list, outfile)



def data(args):
    directory_path = args.dataset
    output_path = args.output_path

    nodes_dir = directory_path.split('/')[-1]
    if (nodes_dir==''):
        nodes_dir=directory_path.split('/')[-2]

    ned_filename = ""
    if nodes_dir=="nsfnetQueue":
        ned_filename = "/Network_nsfnetQueue.ned"
    elif nodes_dir=="geant2bwQueue":
        ned_filename = "/Network_geant2bw.ned"

    global_counter = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".tar.gz"):
            tar = tarfile.open(directory_path+filename, "r:gz")
            dir_info = tar.next()
            print(filename)
            if (not dir_info.isdir()):
                print("Tar file with wrong format")
                sys.exit(0)

            delay_file = tar.extractfile(dir_info.name + "/delayGlobal.txt")
            routing_file = tar.extractfile(dir_info.name + "/Routing.txt")
            ned_file = tar.extractfile(dir_info.name + ned_filename)

            path_train = output_path + 'Dataset_q_size/train'
            path_eval = output_path + 'Dataset_q_size/eval'

            if not os.path.exists(path_train):
                os.makedirs(path_train)

            if not os.path.exists(path_eval):
                os.makedirs(path_eval)

            x = random.uniform(0, 1)
            if x <= 0.8:
                path = path_train

            else:
                path = path_eval

            make_jsons(ned_file,routing_file,delay_file)
            tar = tarfile.open(path + "/sample_" + str(global_counter) + ".tar.gz", "w:gz")

            tar.add('data.txt')
            tar.close()
            os.remove('data.txt')
            global_counter +=1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrating tool from a raw dataset to a valid JSON version.')

    parser.add_argument('--dataset', type=str,help='Path to find the dataset')
    parser.add_argument('--output_path', help='Path where the resulting JSON dataset will be saved', type=str)
    parser.set_defaults(func=data)

    args = parser.parse_args()
    args.func(args)

