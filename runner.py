
import random
from scipy.special import binom
from math import *

import networkx as nx
from networkx.algorithms.approximation import clique


from blincodes.codes import rm
from blincodes import matrix
from blincodes import vector
from blincodes.codes import tools

r = 2
m = 5

cij = [[0 for x in range (2**m - 2**(m - r))] for y in range (2**m - 2**(m - r))]

def gauss_by_min_weighted_row (irmc, codeword_support):
	
	rmcga = irmc.gaussian_elimination(codeword_support)

	print("rmcga:")

	print(rmcga)

	to_remove = []

	j = 0

	for row in rmcga:
		if len([value for value in row.support if value in codeword_support]):
			to_remove.append(j)
		j+=1

	rmcgar = rmcga.submatrix(codeword_support, True).T.submatrix(to_remove, True).T

	return rmcgar

def gauss_codeword_support_with_weight(irmc, desired_weight = 2**(m - r)):

	print("Desired weight", desired_weight)

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight == desired_weight:
			return vec.support

	print("Bad weight")


def gauss_codewords_supports_with_weight_in_range (irmc, eps):


	desired_weight_min = 2**(m - r)
	desired_weight_max = floor(float(2**(m - 2*r + 1)*(2**r - 1))*eps)

	print("Desired weight", desired_weight_min, desired_weight_max)

	codewords_supports = []

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight >= desired_weight_min and vec.hamming_weight <= desired_weight_max:
			codewords_supports.append(vec.support)
	print("Found", len(codewords_supports))

	print("Bad weight")

	return codewords_supports

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

	codewords_supports = gauss_codewords_supports_with_weight_in_range(rmcgac, eps)

	word_len = 2**m - 2**(m - r)

	print("Supports len:", len(codewords_supports))

	for i in range(0, word_len):
		for j in range(i + 1, word_len):
			word_num = 0
			#cij[i][j] = 0
			for word_support in codewords_supports:
				if (i in word_support) and (j in word_support):
					#print("i",i,"j",j,"support",word_support, "word number:", word_num)
					cij[i][j] += 1

				word_num += 1

	G = nx.Graph()
	G.add_nodes_from(range(0, word_len))

	try_it = 0

	while (True):

		print("try it ", try_it)

		#c = ceil( L * (2**(m - r + 1) - 2**r) / (2**(m - r) - 1))

		c= 10

		for i in range(0, word_len):
			for j in range(i + 1, word_len):
				if cij[i][j]*(try_it + 1) > c:
					#print("Add",i,",",j)
					G.add_edge(i,j)

		good_cliques = get_good_cliques(G)

		if len(good_cliques):
			return good_cliques
		else:
			L += step_L
			
		try_it += 1
		c+= 10

def get_b(pbk):
	B = []

	desired_b_size = 0

	for i in range (0, r - 1):
		desired_b_size += binom(m, i)

	print ("Desired size:", desired_b_size)


	while (True):
		codeword_support = gauss_codeword_support_with_weight(pbk)

		pbkc = gauss_by_min_weighted_row(pbk, codeword_support)

		fs_supports = inner_algo(pbkc, 100)

		print("Fs supports:", fs_supports)

		print("Codeword supports:", codeword_support)		

		fs =[]

		for f_support in fs_supports:
			tmp_support = f_support
			tmp_support.extend(codeword_support)

			fs.append(vector.from_support(2**m, tmp_support))

		print("fs:", fs)

		print("B:", B)

		Bm = tools.union(matrix.Matrix(B), matrix.Matrix(fs))

		print(Bm)

		#if len(B) == desired_b_size:
		return Bm





rmc = rm.generator(r,m)
# Secret key generation
M = matrix.nonsingular(rmc.nrows)
tmp_P = [x for x in range(rmc.ncolumns)]
random.shuffle(tmp_P)
P = matrix.permutation(tmp_P)
pubkey = M * rmc * P

print (pubkey)

b = get_b(pubkey)
