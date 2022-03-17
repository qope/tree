from copy import copy
import graphviz as gv
import numpy as np
from scipy.linalg import eigh

def copy_tree(tree):
    new_tree = []
    for node in tree:
        new_tree.append(copy(node))
    return new_tree
def search_above(node_id, Tree):
    for node in Tree:
        if node_id in node["downs"]:
            return node
def tree_dict(m):
    Trees_dict = {}
    node1 = {"id":"s1", "downs":set([])}
    node2 = {"id":"s2", "downs":set([])}
    root_node ={"id":"1", "downs":set([node1["id"], node2["id"]])}
    Trees_dict[2]  = [[root_node, node1, node2],]
    for n in range(3, m+1):
        corrent_Trees = Trees_dict[n-1]
        next_Trees = []
        for Tree in corrent_Trees:
            # print(search_above(node1["id"], Tree))
            for node in Tree:
                next_Tree = copy_tree(Tree)
                new_sigma = {"id":"s"+str(n), "downs":set([])}
                new_node =  {"id":str(n-1), "downs":set([new_sigma["id"], node["id"]])}
                above_node = copy(search_above(node["id"], next_Tree))
                if above_node != None:
                    for i in next_Tree:
                        if i["id"]==above_node["id"]:
                            i["downs"] = (above_node["downs"] - set([node["id"]])) | set([new_node["id"]])
                next_Tree.extend([new_sigma, new_node])
                next_Trees.append(next_Tree)
        Trees_dict[n] = next_Trees
    return Trees_dict
def is_root(node, tree):
    for i in tree:
        if node["id"] in i["downs"]:
            return False
    return True
def is_ok(tree):
    for node in tree:
        if len(node["downs"])==0 and is_root(search_above(node["id"], tree), tree):
            return False
    # print("a")
    return True
def remove_tree(trees):
    new_trees = []
    for i, tree in enumerate(trees):
        if is_ok(tree):
            new_trees.append(tree) 
    return new_trees
def f2(n):
    assert n%2==1 and n>0
    if n==1:
        return 1
    else:
        return n*f2(n-2)
def graph(tree):
    G = gv.Graph()
    for node in tree:
        if len(node["downs"])==0:
            G.node(node["id"], shape="circle")
        else:
            if is_root(node, tree):
                G.node(node["id"], label="", shape="diamond")
            else:
                G.node(node["id"], label="", shape="triangle")
            for i in node["downs"]:
                G.edge(node["id"], i)
    return G
def find_node_by_id(node_id, tree):
    for node in tree:
        if node["id"]==node_id:
            return node
def all_spins(node, tree):
    sm = []
    if len(node["downs"])==0:
        return [node["id"]]
    else:
        for i in node["downs"]:
            sm = sm+all_spins(find_node_by_id(i, tree), tree)
    return sm
def entropy(rho):
    rho2 = (rho+np.transpose(rho))/2.0
    w, v = eigh(rho)
    s = 0
    for i in w:
        s = s - i*np.log2(i) if i > 10**-10 else s
    return s
def bond_entropy(l, l_all, C, d, L):
    permu_list = [l_all.index(i) for i in l+list(set(l_all)-set(l))]
    Cr = C.reshape([d]*L)
    Ct = np.transpose(Cr, permu_list)
    Ct = Ct.reshape(d**len(l), -1)
    rho = Ct@np.transpose(Ct) if Ct.shape[0] < Ct.shape[1] else np.transpose(Ct)@Ct
    return entropy(rho)
def get_root(tree):
    for node in tree:
        if is_root(node, tree):
            return node
def graph_entropy(tree, C, d, L):
    l_all = []
    S = []
    root = get_root(tree)
    flag = 0
    for node in tree:
        if len(node["downs"])==0:
            l_all.append(node["id"])
    for node in tree:
        if len(node["downs"])!=0 and node["id"]!=root["id"]:
            if node["id"] in root["downs"]:
                if flag==0:
                    l = all_spins(find_node_by_id(node["id"], tree), tree)
                    S.append(bond_entropy(l, l_all, C, d, L))
                    flag=1
            else:
                l = all_spins(find_node_by_id(node["id"], tree), tree)
                S.append(bond_entropy(l, l_all, C, d, L))
    return S
def graph_with_entropy(tree, C, d, L):
    l_all = []
    for node in tree:
        if len(node["downs"])==0:
            l_all.append(node["id"])
    G = gv.Graph()
    for node in tree:
        if len(node["downs"])==0:
            G.node(node["id"], shape="circle")
        else:
            if is_root(node, tree):
                G.node(node["id"], label="", shape="diamond")
            else:
                G.node(node["id"], shape="", label="")
            for i in node["downs"]:
                l = all_spins(find_node_by_id(i, tree), tree)
                S =bond_entropy(l, l_all, C, d, L)
                G.edge(node["id"], i, label= str(round(S, 3)))
    return G
