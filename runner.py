
import random

from math import *

import networkx as nx
from networkx.algorithms.approximation import clique


from blincodes.codes import rm
from blincodes import matrix
from blincodes import vector
from blincodes.codes import tools

r = 2
m = 5



cij = [[0] *  (2**m - 2**(m - r))] * (2**m - 2**(m - r))

def gauss_by_min_weighted_row (rmc):


	k = rmc.nrows
	n = rmc.ncolumns


	max_tries = 250
	desired_weight = 2**(m - r) - 1

	for i in range(0, max_tries):
		print ("[ ]\ttry ", i)
		sample = random.sample(range(0, n), k)

		#ssample = sorted(sample)
		#print(ssample, "Len:", len(ssample))

		rmcg = rmc.gaussian_elimination(sample)

		rmcgr = rmcg.submatrix(sample, True)

		
		j = 0
		for row in rmcgr:
			
			if row.hamming_weight == desired_weight:

				#print ("Row", j)

				rmcgrow_ones = rmcg[j].support

				#print ("Ones:", rmcgrow_ones)

				rmcga = rmcg.gaussian_elimination(rmcgrow_ones)

				#print (rmcga)

				rmcgac = rmcga[len(rmcgrow_ones):].submatrix(rmcgrow_ones, True)

				return rmcgac

			j+=1
	print("[-]\tout of tries while doing gauss_by_min_weighted_row")

def gauss_codewords_supports_with_weight_in_range (rmc, M, eps):

	k = rmc.nrows
	n = rmc.ncolumns


	max_tries = 50000
	desired_weight_min = 2**(m - r) - 1
	desired_weight_max = floor(float(2**(m - 2*r + 1)*(2**r - 1))*eps) - 1

	codewords_supports = []

	for i in range(0, max_tries):
		#print ("[ ]\ttry to find codewords_supports", i, "found", len(codewords_supports), "need", M)
		sample = random.sample(range(0, n), k)



		#ssample = sorted(sample)
		#print(ssample, "Len:", len(ssample))

		rmcg = rmc.gaussian_elimination(sample)

		rmcgr = rmcg.submatrix(sample, True)

		j = 0
		for row in rmcgr:
			
			if row.hamming_weight >= desired_weight_min and row.hamming_weight <= desired_weight_max:
				if rmcg[j].support not in codewords_supports:
					codewords_supports.append(rmcg[j].support)
				

				if len(codewords_supports) == M:
					return codewords_supports

			j+=1
	print("[-]\tout of tries while searching for codewords_supports")

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

	desired_clique_size = 3
	desired_number_of_cliques = 5

	G = Gi.copy()


	result_cliques = []

	while (True):

		cliques = nx.find_cliques(G)

		lol = True

		for clique in cliques:
			clique_len = len(clique)
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



		if is_graph_ok(G):
			return get_good_cliques(G)
		else:
			L += step_L

			
		try_it += 1			







rmc = rm.generator(r,m)

rmcgac = gauss_by_min_weighted_row(rmc)

#print("Result:")
print(rmcgac)

result_cliques = inner_algo(rmcgac, 5)

print(result_cliques)
