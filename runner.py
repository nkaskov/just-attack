
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

already_choosen_codeword_supports = []

def gauss_codeword_support_with_weight(irmc, desired_weight = 2**(m - r)):

	print("[i]\tDesired weight of codeword:", desired_weight)

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight == desired_weight and vec.support not in already_choosen_codeword_supports:
			already_choosen_codeword_supports.append(vec.support)
			print("[i]\tchoosen codeword:", vec.support)
			return vec.support

	print("[-]\tBad weight recieved while searching of minimal codeword")

def gauss_codewords_supports_with_weight_in_range (irmc, eps):


	desired_weight_min = 2**(m - r)
	desired_weight_max = floor(float(2**(m - 2*r + 1)*(2**r - 1))*eps)

	print("[i]\tDesired weight range:", desired_weight_min, desired_weight_max)

	codewords_supports = []

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight >= desired_weight_min and vec.hamming_weight <= desired_weight_max:
			codewords_supports.append(vec.support)
	print("[i]\tFound", len(codewords_supports), "codewords supports in range")

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
			print("[i]\tChecking",clique)
			if clique_len >= desired_clique_size:
				if not clique_len % desired_clique_size:

					for i in range(0, int(clique_len / desired_clique_size)):
						result_cliques.append(clique[i*desired_clique_size: (i + 1)*desired_clique_size])

					G.remove_nodes_from(clique)
					lol= False
					break

		if lol:
			#print ("[-]\tThere are some bad cliques")
			return []

		if len(result_cliques) == desired_number_of_cliques:
			print("[+]\tGraph consist only of good cliques")
			return result_cliques

def inner_algo(rmcgac, L):
	

	step_L = 1

	eps = float(sqrt(1 - 1/(2**(m - 2*r + 1))))

	codewords_supports = gauss_codewords_supports_with_weight_in_range(rmcgac, eps)

	word_len = 2**m 

	G = nx.Graph()
	G.add_nodes_from(range(0, word_len))


	cij = [[0 for x in range (word_len)] for y in range (word_len)]

	for i in range(0, word_len):
		for j in range(i + 1, word_len):
			word_num = 0
			#cij[i][j] = 0
			for word_support in codewords_supports:
				if (i in word_support) and (j in word_support):
					#print("i",i,"j",j,"support",word_support, "word number:", word_num)
					cij[i][j] += 1

				word_num += 1

	cij_values = set()

	for i in range(0, word_len):
		for j in range(i + 1, word_len):
			cij_values.add(cij[i][j])

	#print("[i]\tcij values:", cij_values)

	c = max(cij_values)

	print("[i]\tThreshold c choosen to", c)

	for i in range(0, word_len):
		for j in range(i + 1, word_len):
			if cij[i][j] >= c:
				#print("Add",i,",",j)
				G.add_edge(i,j)

	good_cliques = get_good_cliques(G)

	if len(good_cliques):
		return good_cliques
	else:
		print("[-]\tSomething very bad happened")
		
def get_b(pbk):

	B = matrix.Matrix()

	desired_b_size = 0

	for i in range (0, r):
		desired_b_size += binom(m, i)

	print ("[i]\tDesired B size:", desired_b_size)

	try_it = 0

	while (True):

		try_it+=1

		codeword_support = gauss_codeword_support_with_weight(pbk)

		pbkc = tools.truncate(pbk, codeword_support)

		fs_supports = inner_algo(pbkc, 100)

		print("[i]\tFs supports:", fs_supports)

		print("[i]\tCodeword supports:", codeword_support)		

		fs =[]

		for f_support in fs_supports:
			tmp_support = f_support
			tmp_support.extend(codeword_support)

			fs.append(vector.from_support(2**m, tmp_support))

		#print("[i]\tfs vectors after extending:", fs)

		B = tools.union(B, matrix.from_vectors(fs))

		print("[i]\tB with fs vectors:")
		print(B)

		if B.nrows == desired_b_size:
			print("[+]\tB of desired size found on try", try_it)
			return B

def mult(B1, B2):
	rows = []

	for row1 in B1:
		for row2 in B2:
			rows.append(row1&row2)

	return matrix.from_vectors([row for row in matrix.from_vectors(rows).gaussian_elimination() if len(row.support)])


def get_two_basis_code(B1, B2):
	return mult(B1.orthogonal, B2).orthogonal


def get_random_pubkey():

	rmc = rm.generator(r,m)
	# Secret key generation
	M = matrix.nonsingular(rmc.nrows)
	tmp_P = [x for x in range(rmc.ncolumns)]
	random.shuffle(tmp_P)
	P = matrix.permutation(tmp_P)
	pubkey = M * rmc * P

	return pubkey

#print (pubkey)


pubkey1 = get_random_pubkey()
#b = get_b(pubkey)
pubkey2 = get_random_pubkey()

print("pubkey 1:\n", pubkey1)

print("pubkey 2:\n", pubkey2)

print("result:\n", mult(pubkey1, pubkey2))


