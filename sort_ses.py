import numpy as np 
from functions import *

d = 2
L = 12
C = create_free_hopping_state(L)

Tree1, Indices1, sigma_temp1 = make_tree(d, L, C, mps=False)
simple_tree1 = simplify_tree(Tree1, Indices1, sigma_temp1)
graph(simple_tree1).render("a", view=True)
result1 = sorted(graph_entropy(simple_tree1, C, d, L))[::-1]
print(sum(result1))
print(result1)

Tree2, Indices2, sigma_temp2 = make_tree(d, L, C, mps=True)
simple_tree2 = simplify_tree(Tree2, Indices2, sigma_temp2)
result2 = sorted(graph_entropy(simple_tree2, C, d, L))[::-1]
print(sum(result2))
print(result2)

ls = [1,2,3,4,5,7]
for i in range(20):
    goal_node_id = str(ls[np.random.randint(6)])
    for node in Tree:
        if node.node_id==goal_node_id:
            goal_node = node
    Indices = root_on_node(Tree, Indices, goal_node)
    Indices = re_tree(Tree, Indices, d, T=0)
simple_tree = simplify_tree(Tree, Indices, sigma_temp)
G = graph_with_entropy(simple_tree, C, d, L)


def trace(indices, group, rho):
	l = [i for i, x in enumerate(indices)]
	l1 = l[:int(len(l)/2)]
	indices1 = indices[:int(len(indices)/2)]
	# l2 = indices[int(len(indices)/2):]
	int_group1 = [indices1.index(i) for i in group]
	int_comp_group1 = list(set(l1) - set(int_group1))
	permutation_list1 = list(int_group1) + int_comp_group1
	permutation_list2 = [int(len(l)/2) + i for i in permutation_list1]
	permutation_list = permutation_list1 + permutation_list2
	rho = np.transpose(rho.reshape([2]*len(l)), permutation_list)
	rho = rho.reshape([2**len(int_group1), 2**len(int_comp_group1), 2**len(int_group1), 2**len(int_comp_group1)])
	rho = np.transpose(rho, [0,2,1,3])
	rho = rho.reshape([2**(2*len(int_group1)), -1])
	rho = np.eye(2**len(int_group1)).reshape(-1)@rho
	comp_group = [indices[i] for i in int_comp_group1]
	new_indices = comp_group + ["p"+i for i in comp_group]
	return rho.reshape([2**len(comp_group), -1]), new_indices
def print_norm(U, V):
	print(np.linalg.norm(U.reshape(-1)-V.reshape(-1)))

def find_first_cut(indices, C):
	l = [i for i, x in enumerate(indices)]
	min_S = float("inf")
	for group in powerset(l):
		comp_group = list(set(l) - set(group))
		permutation_list = list(group) + comp_group
		C = np.transpose(C.reshape([2]*len(l)), permutation_list).reshape([2**len(group), -1])
		if C.shape[0] < C.shape[1]:
			rho = C@C.T
		else:
			rho = C.T@C
		S = entropy(rho)
		if S < min_S:
			min_S = S
			min_group = group
	group = [x for i, x in enumerate(indices) if i in min_group]
	comp_group = comp_group = list(set(indices) - set(group))
	pair = [trace_v(indices, group, C), trace_v(indices, comp_group, C)]
	return pair

def find_cut(indices, rho):
	indices1 = indices[:int(len(indices)/2)]
	min_S = float("inf")
	S_rho = entropy(rho)
	for group in powerset(indices1):
		comp_group = list(set(indices1) - set(group))
		rho_A, indices_A = trace(indices, list(group), rho)
		rho_B, indices_B = trace(indices, comp_group, rho)
		I = entropy(rho_A) + entropy(rho_B) - S_rho
		if I < min_S:
			min_S = I
			pair = [(rho_A, indices_A), (rho_B, indices_B)]
	print(min_S)
	return pair

pair = find_first_cut(indices, C)
rho_A, indices_A = pair[0]
rho_B, indices_B = pair[1]

history = [(indices_A, indices_B)]
targets = pair
while targets:
	rho, indices = targets.pop()
	if len(indices) > 4:
		pair = find_cut(indices, rho)
		rho_A, indices_A = pair[0]
		rho_B, indices_B = pair[1]
		history.append((indices_A, indices_B))
		targets.extend(pair)
print(history)
# find_cut(indices, rho)
# rho, new_indices = trace(new_indices, ["8", "10"], rho)
# rho, new_indices = trace(new_indices, ["4"], rho)

# rho2, new_indices2 = trace_v(indices, ["1", "2", "4", "8", "10"], C)
# print_norm(rho,rho2)


