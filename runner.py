
import random

from math import *

import networkx as nx
from networkx.algorithms.approximation import clique


from blincodes.codes import rm
from blincodes import matrix
from blincodes import vector
from blincodes.codes import tools

r = 2
m = 7



cij = [[0] *  (2**m - 2**(m - r))] * (2**m - 2**(m - r))

def gauss_by_min_weighted_row (irmc, codeword_support):
	
	rmcga = irmc.gaussian_elimination(codeword_support)

	print("rmcga:")

	print(rmcga)

	rmcgac = rmcga[len(codeword_support):].submatrix(codeword_support, True)

	return rmcgac



def gauss_codeword_support_with_weight(irmc, desired_weight = 2**(m - r)):

	print("Desired weight", desired_weight)

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight == desired_weight:
			return vec.support

	print("Bad weight")


def gauss_codewords_supports_with_weight_in_range (irmc, M, eps):


	desired_weight_min = 2**(m - r) - 1
	desired_weight_max = floor(float(2**(m - 2*r + 1)*(2**r - 1))*eps) - 1

	print("Desired weight", desired_weight_min, desired_weight_max)

	codewords_supports = []

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight >= desired_weight_min and vec.hamming_weight <= desired_weight_max:
			codewords_supports.append(vec.support)
			print("Found", len(codewords_supports), "from", M)
			if len(codewords_supports) == M:
				return codewords_supports

	print("Bad weight")

def is_graph_ok(Gi):

	desired_clique_size = 2**(m - r)
	desired_number_of_cliques = 2**r - 1

	G = Gi.copy()


	good_cliques = 0

	while (True):

		cliques = nx.find_cliques(G)

		lol = True

		for clique in cliques:
			clique_len = len(clique)
			#print("Checking",clique)
			if clique_len >= desired_clique_size:
				if not clique_len % desired_clique_size:
					good_cliques += clique_len / desired_clique_size
					#print("Found good cliques", good_cliques)
					G.remove_nodes_from(clique)
					lol= False
					break

		if lol:
			print ("[ ]\tThere are some bad cliques")
			return False

		if good_cliques == desired_number_of_cliques:
			print("[+]\tGraph consist only of good cliques")
			return True

def get_good_cliques(Gi):

	desired_clique_size = 2**(m - r)
	desired_number_of_cliques = 2**r - 1

	G = Gi.copy()


	result_cliques = []

	while (True):

		cliques = nx.find_cliques(G)

		lol = True

		for clique in cliques:
			clique_len = len(clique)
			print("Checking",clique)
			if clique_len >= desired_clique_size:
				if not clique_len % desired_clique_size:

					for i in range(0, int(clique_len / desired_clique_size)):
						result_cliques.append(clique[i*desired_clique_size: (i + 1)*desired_clique_size])

					G.remove_nodes_from(clique)
					lol= False
					break

		if lol:
			print ("[-]\tThere are some bad cliques")
			return []

		if len(result_cliques) == desired_number_of_cliques:
			print("Graph consist only of good cliques")
			return result_cliques

def inner_algo(rmcgac, L):
	

	step_L = 1

	eps = float(sqrt(1 - 1/(2**(m - 2*r + 1))))

	word_len = 2**m - 2**(m - r)

	G = nx.Graph()
	G.add_nodes_from(range(0, word_len))

	try_it = 0

	while (True):

		print("try it ", try_it)

		M = L * 2**(2*r - 1)

		print("M", M, "\n")
		#c = ceil( L * (2**(m - r + 1) - 2**r) / (2**(m - r) - 1))
		c = 50

		codewords_supports = gauss_codewords_supports_with_weight_in_range(rmcgac, M, eps)



		for i in range(0, word_len):
			for j in range(i + 1, word_len):
				for word_support in codewords_supports:
					if i in word_support and j in word_support:
						cij[i][j] += 1
						if cij[i][j] > c:
							#print("Add",i,",",j)
							G.add_edge(i,j)

		good_cliques = get_good_cliques(G)

		if len(good_cliques):
			return good_cliques
		else:
			L += step_L

			
		try_it += 1			







rmc = rm.generator(r,m)
# Secret key generation
M = matrix.nonsingular(rmc.nrows)
tmp_P = [x for x in range(rmc.ncolumns)]
random.shuffle(tmp_P)
P = matrix.permutation(tmp_P)
pubkey = M * rmc * P

print (pubkey)

#rmcgac = gauss_by_min_weighted_row(pubkey)
'''
#print("Result:")
print(rmcgac)

result_cliques = inner_algo(rmcgac, 5)

print(result_cliques)
'''


codeword_support = gauss_codeword_support_with_weight(pubkey)
print(codeword_support)

print(gauss_by_min_weighted_row(pubkey, codeword_support))