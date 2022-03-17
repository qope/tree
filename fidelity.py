from functions import *
from scipy.sparse.linalg import svds
import pandas as pd

def fidelity_of_two_states(Tree, sigma_temp, C):
	f = 0.0
	for i in range(d**L):
		spins = list(map(int, list(format(i, "0"+str(L)+"b"))))
		f += check_product_function(C, list(spins))*catch_val(Tree, spins, sigma_temp)
	return f

d = 2
L = 12
tree = []
mps = []
for chi in range(1, 50, 5):
	C = np.array([random.random() for i in range(d**L)])
	C = C/np.sqrt(C@np.transpose(C))
	Tree, Indices, sigma_temp = make_tree_with_chi(d, L, C, chi=chi)
	tree.append(fidelity_of_two_states(Tree, sigma_temp, C))
	Tree, Indices, sigma_temp = make_tree_with_chi(d, L, C, chi=chi, mps=True)
	mps.append(fidelity_of_two_states(Tree, sigma_temp, C))

df = pd.DataFrame({"chi":list(range(1, 50, 5)), "tree":tree, "mps":mps})
df.to_csv("a.csv")


