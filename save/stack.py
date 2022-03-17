import random
import numpy as np
from itertools import combinations
from scipy.linalg import svd, eigh
from copy import copy
class c_Node:
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        assert check_c_node(self)

class Node:
    def __init__(self, data_type, data, down_l, down_r, up=None):
        self.node_id = random.randint(0, 10**10)
        self.data_type = data_type 
        self.data = data
        self.up = up
        self.down_l = down_l
        self.down_r = down_r
        self.tag_down_l = down_l.index_type
        self.tag_down_r = down_r.index_type
        # print(self.up)
        assert check_node(self)

class Index:
    def __init__(self,index_type,index_dim):
        self.index_id = random.randint(0, 10**10)
        self.index_type = index_type
        self.index_dim = index_dim
def search_node_by_index(index, tree):
    for node in tree:
        if node.up != None:
            if index.index_id == node.up.index_id:
                return node
def check_c_node(c_node):
    dim = 1
    for index_dict in c_node.indices:
        dim = dim*index_dict["index"].index_dim
    return dim==c_node.data.size
def check_node(Node):
    data_size = Node.data.size
    down_size = Node.down_l.index_dim*Node.down_r.index_dim
    # print(down_size)
    up_size = 1 if Node.up==None else Node.up.index_dim
    return data_size==up_size*down_size
def remove_index(index_id, Indices):
    return [i for i in Indices if i.index_id!=index_id]
def find_root_node(Tree):
    for node in Tree:
        if node.data_type=="diagonal":
            return node

import graphviz as gv # needs graphviz to run the plot function
def find_side_node(index, Tree, s_dict, n_dict):
    if index.index_type=="sigma":
        down_node_name = s_dict[index.index_id]
    for i in Tree:
        if i.up != None:
            if i.up.index_id==index.index_id:
                down_node_name = n_dict[i.node_id]
        if index.index_id in [i.down_l.index_id, i.down_r.index_id]:
            up_node_name = n_dict[i.node_id]
    # print(up_node_name, down_node_name)
    return up_node_name, down_node_name

def get_downs(index, node_dict, Tree):
    if index.index_type == "sigma":
        return node_dict[index.index_id]
    else:
        for node in Tree:
            if node.up != None:
                if node.up.index_id == index.index_id:
                    return node_dict[node.node_id]
def simplify_tree(Tree, Indices, sigma_temp):
    v_dict = {index.index_id:"v{}".format(i) for i,index in enumerate(Indices)} # naming virtual indices as order
    s_dict = {index.index_id:"s{}".format(i+1) for i,index in enumerate(sigma_temp)} # naming sigma indices
    n_dict = {node.node_id:str(i) for i,node in enumerate(Tree)}
    node_dict = {**s_dict, **n_dict} # merge v_dict and s_dict
    simple_tree = []
    for node in Tree:
        simple_tree.append({"id":n_dict[node.node_id], "downs":set([get_downs(node.down_l, node_dict, Tree),\
         get_downs(node.down_r, node_dict, Tree)])})
    for sigma in sigma_temp:
        simple_tree.append({"id":s_dict[sigma.index_id], "downs":{}})
    return simple_tree
def plot(Tree, Indices, sigma_temp):
    v_dict = {index.index_id:"v{}".format(i) for i,index in enumerate(Indices)} # naming virtual indices as order
    s_dict = {index.index_id:"s{}".format(i) for i,index in enumerate(sigma_temp)} # naming sigma indices
    n_dict = {node.node_id:str(i) for i,node in enumerate(Tree)}
    index_dict = {**v_dict, **s_dict} # merge v_dict and s_dict
    G = gv.Graph()
    for i in range(len(sigma_temp)):
        G.node("s{}".format(i), shape="circle")
    for i, n in enumerate(Tree):
        if n.up!=None:
            G.node(str(i), shape="triangle", label="")
        else:
            G.node(str(i), shape="diamond", label="")
    for i in Indices+sigma_temp:
        up_node_name, down_node_name = find_side_node(i, Tree, s_dict, n_dict)
        # print(down_node_name, up_node_name)
        G.edge(up_node_name, down_node_name)
    return G
def check_product_function(C, ss):
    summ = 0
    while ss:
        summ = summ*2
        summ += ss.pop(0)
    return C[summ]
def entropy(rho):
    rho2 = (rho+np.transpose(rho))/2.0
    w, v = eigh(rho)
    assert np.abs(np.sum(w)-1) < 10**-10
    s = 0
    for i in w:
        s = s - i*np.log2(i) if i > 10**-10 else s
    return s
def find_pair(c_node, d):
    I_max = 0.0
    pair = (c_node.indices[0], c_node.indices[1])
    dim_l = [index_dict["index"].index_dim for index_dict in c_node.indices]
    for a_index, b_index in list(combinations(c_node.indices, 2)):
        l = list(range(len(c_node.indices)))
        l[0],l[a_index["place"]] = l[a_index["place"]], l[0]
        l[1],l[b_index["place"]] = l[b_index["place"]],l[1]
        c_temp = c_node.data.reshape(dim_l)
        c_temp = np.transpose(c_temp, l)
        c_AB = c_temp.reshape(d**2, -1)
        c_A = c_temp.reshape(d, -1)
        c_B = np.transpose(c_A).reshape(d, -1)
        rho_AB = c_AB@np.transpose(c_AB)
        rho_A = c_A@np.transpose(c_A)
        rho_B = c_B@np.transpose(c_B)
        I = entropy(rho_A)+entropy(rho_B)-entropy(rho_AB)
        if I>I_max:
            # print(I_max)
            I_max=I
            pair = (a_index, b_index)
    return pair

def make_tree(d, L, C):
    sigma_temp = [Index(index_type= "sigma", index_dim=d) for i in range(L)]
    c_indices = [{"place":i, "index":sigma_temp[i]} for i in range(L)]
    c_node = c_Node(data=C, indices = c_indices)
    Tree = []
    Indices = []

    while 4<=len(c_node.indices):
        assert check_c_node(c_node)
        # print(len(c_node.indices), c_node.data.shape)
        a, b = find_pair(c_node, d)
        # print(len(list(combinations(c_node.indices, 2))))
        assert 4<=len(c_node.indices)
        l = list(range(len(c_node.indices)))
        l[0],l[a["place"]] = l[a["place"]], l[0]
        l[1],l[b["place"]] = l[b["place"]],l[1]
        permute_list = l
        # print(permute_list)
        split_list = [c_node.indices[i]["index"].index_dim for i in range(len(c_node.indices))]
        p = c_node.data.reshape(split_list)
        p = np.transpose(p,permute_list).reshape([a["index"].index_dim*b["index"].index_dim,-1])
        U,s,Vh = svd(p,full_matrices=False)
        new_index = Index(index_type="virtual", index_dim = U.shape[1])
        Indices.append(new_index)
        A = Node(data_type="three_leg", data=U.reshape([a["index"].index_dim, b["index"].index_dim,-1]),        up=new_index, down_l=a["index"], down_r=b["index"])
        Tree.append(A)
        rest_c = (np.diag(s)@Vh).reshape(-1)
        assert len(rest_c.shape)==1
        c_new_indices = c_node.indices
        c_new_indices[0], c_new_indices[a["place"]] = c_new_indices[a["place"]], c_new_indices[0]
        c_new_indices[1], c_new_indices[b["place"]] = c_new_indices[b["place"]], c_new_indices[1]
        c_new_indices = [{"place":0, "index":new_index}]+c_new_indices[2:]
        c_new_indices = [{"place":i, "index":c_new_indices[i]["index"]} for i in range(len(c_new_indices))] # label i
        # print(c_new_indices)
        c_node.data = rest_c # store c
        c_node.indices = c_new_indices # store indices
        assert check_c_node(c_node)
    assert len(c_node.indices)==3
    virtual_index = c_node.indices[0]
    assert virtual_index["index"].index_type=="virtual"
    a = c_node.indices[1]
    b = c_node.indices[2]
    p = c_node.data.reshape([virtual_index["index"].index_dim, -1])
    assert c_node.data.size == a["index"].index_dim*b["index"].index_dim*virtual_index["index"].index_dim
    U,s,Vh = svd(p,full_matrices=False)
    new_index1 = Index(index_type="virtual", index_dim = U.shape[1])
    new_index2 = Index(index_type="virtual", index_dim = Vh.shape[0])
    Indices.append(new_index1)
    Indices.append(new_index2)
    S = np.diag(s).reshape([new_index1.index_dim, new_index2.index_dim])
    diagonal_Node = Node(data_type="diagonal", data=S, down_l = new_index1, down_r = new_index2)
    corresponding_node = search_node_by_index(virtual_index["index"], Tree)
    corresponding_node.data  =  corresponding_node.data.reshape([-1,corresponding_node.up.index_dim])@U
    Indices = remove_index(corresponding_node.up.index_id, Indices)
    corresponding_node.up = new_index1
    A = Node(data_type="three_leg", data=np.transpose(Vh.reshape([new_index2.index_dim, a["index"].index_dim,    b["index"].index_dim])),up=new_index2, down_l=b["index"], down_r=a["index"])
    Tree.append(diagonal_Node)
    Tree.append(A)

    return Tree, Indices, sigma_temp


def product(corrent_node, Tree, s_num, n_dict):
    ### if the down nodes were sigma's, substruct the numbers
    if corrent_node.tag_down_l == "sigma":
        U = corrent_node.data.reshape([corrent_node.down_l.index_dim, -1])
        s = s_num[corrent_node.down_l.index_id]
        corrent_node.data = U[s].reshape(-1)
        corrent_node.tag_down_l = "done"
    if corrent_node.tag_down_r == "sigma":
        if corrent_node.tag_down_l=="done":
            U = corrent_node.data.reshape([corrent_node.down_r.index_dim, -1])
            s = s_num[corrent_node.down_r.index_id]
            corrent_node.data = U[s].reshape(-1)
        else:
            U = corrent_node.data.reshape([corrent_node.down_l.index_dim, corrent_node.down_r.index_dim, -1])
            U = np.transpose(U, [1, 0, 2]).reshape(corrent_node.down_r.index_dim, -1)
            s = s_num[corrent_node.down_r.index_id]
            corrent_node.data = U[s].reshape(-1)
        corrent_node.tag_down_r = "done"
    ### if the down nodes were not done nodes, recurse this function
    if corrent_node.tag_down_l != "done":
        product(search_node_by_index(corrent_node.down_l, Tree), Tree, s_num, n_dict)
    if corrent_node.tag_down_r != "done":
        product(search_node_by_index(corrent_node.down_r, Tree), Tree, s_num, n_dict)
    ### if the both down nodes had been done, contruct with the above tensor
    assert corrent_node.tag_down_l == "done" and corrent_node.tag_down_r == "done"
    if corrent_node.data_type != "diagonal":
        for node in Tree:
            if node.down_l.index_id == corrent_node.up.index_id:
                U = node.data
                U = U.reshape([node.down_l.index_dim, -1])
                node.data = (np.transpose(corrent_node.data.reshape(-1))@U).reshape([node.down_r.index_dim, -1])
                node.tag_down_l = "done"
            if node.down_r.index_id == corrent_node.up.index_id:
                assert node.tag_down_l == "done" # l has been searched already
                U = node.data.reshape([node.down_r.index_dim, -1])
                node.data = (np.transpose(corrent_node.data.reshape(-1))@U).reshape(-1)
                node.tag_down_r = "done"
def move_root(Tree, Indices, direction):
    root_node = find_root_node(Tree)
    a = root_node.down_l
    b = root_node.down_r
    if direction=="ll":
        next_node = search_node_by_index(root_node.down_l, Tree)
        assert next_node.down_l.index_type != "sigma"
        u = next_node.up
        l = next_node.down_l
        r = next_node.down_r
        p = next_node.data@root_node.data
        p = p.reshape([next_node.down_l.index_dim, -1])
        U, s, Vh = svd(p, full_matrices=False)
        index1 = Index(index_type="virtual", index_dim = U.shape[1])
        index2 = Index(index_type="virtual", index_dim = Vh.shape[0])
        Indices.append(index1)
        Indices.append(index2)
        down_node = search_node_by_index(l, Tree)
        W = down_node.data
        down_node.data = W@U
        down_node.up = index1
        Indices = remove_index(l.index_id, Indices)
        Indices = remove_index(a.index_id, Indices)
        root_node_id = root_node.node_id
        next_node_id = next_node.node_id
        next_node.node_id = root_node.node_id
        root_node.node_id = next_node_id
        next_node.data_type = "diagonal"
        next_node.data = np.diag(s)
        next_node.up = None
        next_node.down_l = index1
        next_node.down_r = index2

        root_node.data_type = "three_leg"
        root_node.data = np.transpose(Vh)
        root_node.up = index2
        root_node.down_l = r
        root_node.down_r = b
    if direction=="lr": # unchecked
        next_node = search_node_by_index(root_node.down_l, Tree)
        assert next_node.down_r.index_type != "sigma"
        u = next_node.up
        l = next_node.down_l
        r = next_node.down_r
        p = next_node.data@root_node.data
        p = np.transpose(p.reshape([l.index_dim, r.index_dim, b.index_dim]), [1,0,2])
        p = p.reshape([next_node.down_r.index_dim, -1])
        U, s, Vh = svd(p, full_matrices=False)
        index1 = Index(index_type="virtual", index_dim = U.shape[1])
        index2 = Index(index_type="virtual", index_dim = Vh.shape[0])
        Indices.append(index1)
        Indices.append(index2)
        down_node = search_node_by_index(r, Tree)
        W = down_node.data
        down_node.data = W@U
        down_node.up = index1
        Indices = remove_index(r.index_id, Indices)
        Indices = remove_index(a.index_id, Indices)
        root_node_id = root_node.node_id
        next_node_id = next_node.node_id
        next_node.node_id = root_node.node_id
        root_node.node_id = next_node_id
        next_node.data_type = "diagonal"
        next_node.data = np.diag(s)
        next_node.up = None
        next_node.down_l = index1
        next_node.down_r = index2

        root_node.data_type = "three_leg"
        root_node.data = np.transpose(Vh)
        root_node.up = index2
        root_node.down_l = l
        root_node.down_r = b
    if direction=="rr": 
        next_node = search_node_by_index(root_node.down_r, Tree)
        assert next_node.down_r.index_type != "sigma"
        u = next_node.up
        l = next_node.down_l
        r = next_node.down_r
        p = next_node.data@np.transpose(root_node.data) # Actualy, no need to transpose. 
        p = np.transpose(p.reshape([l.index_dim, r.index_dim, b.index_dim]), [1,0,2])
        p = p.reshape([next_node.down_r.index_dim, -1])
        U, s, Vh = svd(p, full_matrices=False)
        index1 = Index(index_type="virtual", index_dim = U.shape[1])
        index2 = Index(index_type="virtual", index_dim = Vh.shape[0])
        Indices.append(index1)
        Indices.append(index2)
        down_node = search_node_by_index(r, Tree)
        W = down_node.data
        down_node.data = W@U
        down_node.up = index1
        Indices = remove_index(r.index_id, Indices)
        Indices = remove_index(b.index_id, Indices)
        root_node_id = root_node.node_id
        next_node_id = next_node.node_id
        next_node.node_id = root_node.node_id
        root_node.node_id = next_node_id

        next_node.data_type = "diagonal"
        next_node.data = np.diag(s)
        next_node.up = None
        next_node.down_l = index1
        next_node.down_r = index2
        root_node.data_type = "three_leg"
        root_node.data = np.transpose(Vh)
        root_node.up = index2
        root_node.down_l = l
        root_node.down_r = a    
    if direction=="rl": 
        next_node = search_node_by_index(root_node.down_r, Tree)
        assert next_node.down_l.index_type != "sigma"
        u = next_node.up
        l = next_node.down_l
        r = next_node.down_r
        p = next_node.data@np.transpose(root_node.data) # Actualy, no need to transpose. 
        p = p.reshape([next_node.down_l.index_dim, -1])
        U, s, Vh = svd(p, full_matrices=False)
        index1 = Index(index_type="virtual", index_dim = U.shape[1])
        index2 = Index(index_type="virtual", index_dim = Vh.shape[0])
        Indices.append(index1)
        Indices.append(index2)
        down_node = search_node_by_index(l, Tree)
        W = down_node.data
        down_node.data = W@U
        down_node.up = index1
        Indices = remove_index(l.index_id, Indices)
        Indices = remove_index(b.index_id, Indices)
        root_node_id = root_node.node_id
        next_node_id = next_node.node_id
        next_node.node_id = root_node.node_id
        root_node.node_id = next_node_id
        next_node.data_type = "diagonal"
        next_node.data = np.diag(s)
        next_node.up = None
        next_node.down_l = index1
        next_node.down_r = index2
        root_node.data_type = "three_leg"
        root_node.data = np.transpose(Vh)
        root_node.up = index2
        root_node.down_l = r
        root_node.down_r = a
    return Indices
def copy_index(index):
    if index!=None:
        i = Index(index_type = index.index_type, index_dim = index.index_dim)
        i.index_id = index.index_id
        return i
def catch_val(Tree, spins, sigma_temp):
    Tree2 = [Node(data_type=node.data_type, data=node.data, up=copy_index(node.up), \
    down_l=copy_index(node.down_l), down_r=copy_index(node.down_r)) for node in Tree] # take deep copy of the Tree
    n_dict = {node.node_id:str(i) for i,node in enumerate(Tree2)}
    root_node = find_root_node(Tree2)
    s_num = {index.index_id:spins[i] for i, index in enumerate(sigma_temp)}
    product(root_node, Tree2, s_num, n_dict)
    (a,) = root_node.data
    return a


if __name__ == "__main__":
    d = 2
    L = 8
    spins = np.random.randint(d, size = L)
    C = np.array([random.random() for i in range(d**L)])
    C = C/np.sqrt(C@np.transpose(C))
    Tree, Indices, sigma_temp = make_tree(d, L, C)
    # plot(Tree, Indices, sigma_temp).render('graph.gv', view=True) 
    print(catch_val(Tree, spins, sigma_temp))
    print(check_product_function(C, list(spins)))
    move_root(Tree, Indices, direction="rr")
    print(catch_val(Tree, spins, sigma_temp))


