

import sys
import tensorflow as tf
from pysmiles import read_smiles, add_explicit_hydrogens
import glob
import numpy as np
import networkx as nx
import json
import os
import random
import tarfile


one_hot_atom = {'H': [1,0,0,0,0],
                'C': [0,1,0,0,0],
                'N': [0,0,1,0,0],
                'O': [0,0,0,1,0],
                'F': [0,0,0,0,1]}

atom_number = { 'H': 1,
                'C': 6,
                'N': 7,
                'O': 8,
                'F': 9}


one_hot_hybridization = {'H': [0,0,0],
                        'C': [0,0,1],
                        'N': [0,0,1],
                        'O': [0,0,1],
                        'F': [0,0,1] }


one_hot_bond = {'1': [1,0,0,0],
                '2': [0,1,0,0],
                '3': [0,0,1,0],
                '4': [0,0,0,1]}



def main():
    path_train = './Dataset_qm9/train'
    path_eval = './Dataset_qm9/eval'

    if not os.path.exists(path_train):
        os.makedirs(path_train)

    if not os.path.exists(path_eval):
        os.makedirs(path_eval)

    samples = glob.glob('/data/qm9/dsgdb9nsd.xyz/*.xyz')

    global_counter = 0
    for s in samples:
        x = random.uniform(0, 1)
        if x <= 0.9:
            path = path_train

        else:
            path = path_eval

        mu_value = process_sample(global_counter, path,s)
        global_counter += 1


def process_sample(global_counter, path, s):
    data = {}
    f = open(s, 'r')

    #num_atoms
    n_atoms = int(f.readline())

    #properties
    tag, index, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G,Cv = f.readline().split()
    data['mu'] = float(mu)
    data['alpha'] = float(alpha)
    data['homo'] = float(homo)
    data['lumo'] = float(lumo)
    data['gap'] = float(gap)
    data['r2'] = float(r2)
    data['zpve'] = float(zpve)
    data['U0'] = float(U0)
    data['U'] = float(U)
    data['H'] = float(H)
    data['G']= float(G)
    data['Cv'] = float(Cv)

    #atoms
    atom_info = [] #at the end we transform this into a one-hot
    data['entities'] = {}


    #create the entities and save the distances
    for i in range(n_atoms):
        data['entities']['a'+str(i)] = 'atom'
        aux = f.readline()
        aux = aux.replace("*^", "e")

        type, pos_x, pos_y, pos_z, partial_charge  = aux.split()
        atom_info.append([type, pos_x, pos_y, pos_z])

    #frequencies
    _ = f.readline()

    # smile
    smiles = f.readline().split()[0]  # only the first formula
    mol = read_smiles(smiles, explicit_hydrogen=True)

    aromatic_info = mol.nodes(data='aromatic')
    charge_info = mol.nodes(data = 'charge')
    charge_info = [v[1] for v in charge_info]    #now we have the charge for each one

    data['acceptor'] = [int(v < 0) for v in charge_info]
    data['donor'] = [int(v > 0) for v in charge_info]


    #create the features
    data['element'] = []
    data['atomic_number'] = []
    data['aromatic'] = [int(v[1]) for v in aromatic_info]
    data['hybridization'] = []

    for i in range(n_atoms):
        element_type = atom_info[i][0]
        element_oh = one_hot_atom[element_type]
        num = atom_number[element_type]
        hy = one_hot_hybridization[element_type]
        data['element'] = data['element'] + element_oh
        data['atomic_number'].append(num)
        data['hybridization'] = data['hybridization'] + hy


    #calculate the adjacencies
    adj = nx.to_numpy_matrix(mol, weight = 'order' )

    data['adj_atom_atom'] = {}  #we need to transform it into a directed graph
    for i in range(n_atoms):
        data['adj_atom_atom']['a'+str(i)] = []

        for j in range(n_atoms):
            if adj[i,j] > 0:   #then they are adjacent
                #find the euclidean distance
                _, x1,y1,z1 = atom_info[i]
                _, x2,y2,z2 = atom_info[j]
                a = np.array([float(x1),float(y1),float(z1)])
                b = np.array([float(x2),float(y2),float(z2)])

                dist = np.linalg.norm(a-b)

                #find the bond one-hot encoding
                bond = one_hot_bond[str(int(adj[i,j]))]
                parameters = [dist] + bond

                data['adj_atom_atom']['a'+str(i)].append(['a'+str(j), parameters])


    result = []
    result.append(data)
    with open('data.json', 'a') as outfile:
        json.dump(result, outfile)

    #write the json file
    tar = tarfile.open(path + "/sample_" + str(global_counter) + ".tar.gz", "w:gz")

    tar.add('data.json')
    tar.close()
    os.remove('data.json')

    return data['mu']







if __name__ == "__main__":
        main ()