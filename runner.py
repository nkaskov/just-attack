
import random

from math import *

import networkx as nx
from networkx.algorithms.approximation import clique


from blincodes.codes import rm
from blincodes import matrix
from blincodes import vector

r = 3
m = 7

rmc = rm.generator(r,m)


k = rmc.nrows
n = rmc.ncolumns

print("[ ]\t", k, 'x', n)

def gauss_by_min_weighted_row ():

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

	max_tries = 5000
	desired_weight_min = 2**(m - r) - 1
	desired_weight_max = floor(float(2**(m - 2*r + 1)*(2**r - 1))*eps) - 1

	codewords_supports = []

	for i in range(0, max_tries):
		print ("[ ]\ttry to find codewords_supports", i, "found", len(codewords_supports), "need", M)
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

def is_graph_ok(G):

	desired_clique_size = 2**(m - r)
	desired_number_of_cliques = 2**r - 1

	all_cliques = nx.enumerate_all_cliques(G)

	print("trying")

	all_elems = {}

	for clique in all_cliques:
		print(len(clique))
		if len(clique) == desired_clique_size:
			all_elems.update(clique)


	print("probably ok")

	for i in range (0, 2**m - 2**(m - r)):
		if not i in all_elems:
			print("Not ok for", i)
			return False
	print ("Ok")
	return True


        

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
		c = ceil( L * (2**(m - r + 1) - 2**r) / (2**(m - r) - 1))

		codewords_supports = gauss_codewords_supports_with_weight_in_range(rmcgac, M, eps)



		for i in range(0, word_len):
			for j in range(i + 1, word_len):
				cij = 0
				for word_support in codewords_supports:
					if i in word_support and j in word_support:
						cij += 1
						if cij > c:
							print("Add",i,",",j)
							G.add_edge(i,j)
							break


		if is_graph_ok(G):
			print("All cliques:")

			cliques = nx.find_cliques(G)


			for clique in cliques:
				print(clique)
			return nx.find_cliques(G)
		else:
			L += step_L

			
		try_it += 1			









rmcgac = gauss_by_min_weighted_row()

#print("Result:")
print(rmcgac)

inner_algo(rmcgac, 100)