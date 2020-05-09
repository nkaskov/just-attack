
import random
from scipy.special import binom
from math import *
import time

import networkx as nx
from networkx.algorithms.approximation import clique


from blincodes.codes import rm
from blincodes import matrix
from blincodes import vector
from blincodes.codes import tools

r = 2
m = 6



def print_log(text, mode = 'info'):

	#print(time.asctime( time.gmtime(time.time())), end='\t')

	if mode == 'info':
		print('[i]', end ='\t')
	elif mode == 'good':
		print('[+]', end ='\t')
	elif mode == 'bad':
		print('[-]', end ='\t')
	else:
		print('!\tBad print log mode')
		return


	print(text)

def gauss_codeword_support_with_weight(irmc, already_choosen_codeword_supports = []):
	desired_weight = 2**(m - r)

	#print("[i]\tDesired weight of codeword:", desired_weight)

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight == desired_weight and vec.support not in already_choosen_codeword_supports:
			already_choosen_codeword_supports.append(vec.support)
			#print("[i]\tchoosen codeword:", vec.support)
			print("[+]\tFound codeword with weight", vec.hamming_weight)
			return vec.support

	print("[-]\tBad weight recieved while searching of minimal codeword")

def gauss_codewords_supports_with_weight_in_range (irmc, eps):


	desired_weight_min = 2**(m - r)
	desired_weight_max = floor(float(2**(m - 2*r + 1)*(2**r - 1))*eps)

	#print("[i]\tDesired weight range:", desired_weight_min, desired_weight_max)

	codewords_supports = []

	for vec in tools.iter_codewords(irmc):
		if vec.hamming_weight >= desired_weight_min and vec.hamming_weight <= desired_weight_max:
			codewords_supports.append(vec.support)

	print("[+]\tFound", len(codewords_supports), "codewords supports in weight range", desired_weight_min, ":", desired_weight_max)

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
			#print("[i]\tChecking",clique)
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

def inner_algo(rmcgac):
	
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
	
	already_choosen_codeword_supports = []

	for i in range (0, r):
		desired_b_size += int(binom(m, i))

	#print ("[i]\tDesired B size:", desired_b_size)

	try_it = 0

	while (True):

		try_it+=1

		codeword_support = gauss_codeword_support_with_weight(pbk, already_choosen_codeword_supports)

		already_choosen_codeword_supports.append(codeword_support)

		pbkc = tools.truncate(pbk, codeword_support)

		fs_supports = inner_algo(pbkc)

		#print("[i]\tFs supports:", fs_supports)

		#print("[i]\tCodeword supports:", codeword_support)

		fs =[]

		for f_support in fs_supports:
			tmp_support = f_support
			tmp_support.extend(codeword_support)

			fs.append(vector.from_support(2**m, tmp_support))

		#print("[i]\tfs vectors after extending:", fs)

		B = tools.union(B, matrix.from_vectors(fs))

		#print("[i]\tB with fs vectors:")
		#print(B)


		if B.nrows == desired_b_size:
			print("[+]\tBasis B of desired size found on try", try_it)
			return B

		print("[+]\tFound", B.nrows,"of", desired_b_size,"basis B vectors on try", try_it)

def mult(B1, B2):
	rows = []

	for row1 in B1:
		for row2 in B2:
			rows.append(row1&row2)

	return matrix.from_vectors([row for row in matrix.from_vectors(rows).gaussian_elimination() if len(row.support)])

def get_two_basis_code(B1, B2):
	return mult(B1.orthogonal, B2).orthogonal

def pubkey_gen():

	rmc = rm.generator(r,m)
	# Secret key generation
	M = matrix.nonsingular(rmc.nrows)
	tmp_P = [x for x in range(rmc.ncolumns)]
	random.shuffle(tmp_P)
	P = matrix.permutation(tmp_P)
	pubkey = M * rmc * P

	return pubkey

def solve_smth(gpub):
	onev = vector.from_support_supplement(2**m)

	return gpub.T.solve(onev)[1]

def build_a(gpub):
	a = solve_smth(gpub)

	#print("[i]\tvector a:", a)

	removing_num = 0

	if len(a.support):
		removing_num = a.support[0]
	else:
		print("[-]\tBad vector a:", a)

	A_rows = [a]

	for i in range(0, m + 1):
		if i != removing_num:
			A_rows.append(a ^ vector.from_support(m + 1, [i]))

	return matrix.from_vectors(A_rows)

def get_perm(gpub):

	agpub = build_a(gpub)*gpub

	#print("[i]\tAGpub:")
	#print(agpub)

	return matrix.permutation([row.value for row in agpub.T.submatrix([0], True)])


def xgcd(a, b):
	#from Wiki
    """return (g, x, y) such that a*x + b*y = g = gcd(a, b)"""
    x0, x1, y0, y1 = 0, 1, 1, 0
    while a != 0:
        (q, a), b = divmod(b, a), a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    return b, x0, y0


def inner_step(gpub, x, y):
	q = ceil((-y)/x)
	s = q*x + y

	#print("q", q, "s", s)

	if s!= 0 and q != 0:

		rm_s = gpub

		for i in range(0,s - 1):
			rm_s = mult(rm_s, gpub)

		rm_qr = gpub

		for i in range(0,q - 1):
			rm_qr = mult(rm_qr, gpub)

		rm_qr = rm_qr.orthogonal

		rm_dm = rm_qr

		for i in range(0, x - 1):
			rm_dm = mult(rm_dm, rm_qr)

		rm_dm = mult(rm_dm, rm_s)

		return rm_dm

	elif s != 0:

		rm_s = gpub

		for i in range(0,s - 1):
			rm_s = mult(rm_s, gpub)
	else:
		rm_qr = gpub

		for i in range(0,q - 1):
			rm_qr = mult(rm_qr, gpub)

		rm_qr = rm_qr.orthogonal

		rm_dm = rm_qr

		for i in range(0, x - 1):
			rm_dm = mult(rm_dm, rm_qr)

		return rm_dm


def step1(gpub):

	g, x, y = xgcd(m - 1, r)

	#print("x", x, "y", y)

	if x == 0 and y == 1:
		return g, gpub
	elif x > 0 and y < 0:
		return g, inner_step(gpub, x, y)
	elif x < 0 and y > 0:
		return g, inner_step(gpub, 1 - x, -y).orthogonal
	else:
		print("[-]\tBad x y")

def step5(gpub, permuted_rm):

	#print("Gpub size", gpub.nrows, gpub.ncolumns)

	#print("permuted size", permuted_rm.nrows, permuted_rm.ncolumns)

	rows = []

	for i in range(0, gpub.nrows):

		rows.append(permuted_rm.T.solve(gpub[i])[1])

	#print("Rows:", rows)
	return matrix.from_vectors(rows)
	#return matrix.Matrix()

def perform_attack(gpub):
	
	step1_gpub = gpub

	global r
	global m

	dual_code = False

	if m <= 2*r:

		dual_code = True

		r = m - 1 - r

		step1_gpub = gpub.orthogonal



	g, rmdm = step1(step1_gpub)

	if g != 1:
		basis = get_b(rmdm)

		rmdm = get_two_basis_code (rmdm, basis)

	#print("RMDM", rmdm)

	if dual_code:
		r = m - 1 - r

	P1 = get_perm(rmdm)

	#print("Permutation:\n", P1)

	permuted_rm = rm.generator(r,m) * P1

	#print("Permuted:\n", permuted_rm)

	M1 = step5(gpub, permuted_rm)

	return M1, P1

def check_attack(gpub, M1, P1):

	print ("Pubkey:\n", gpub)

	pubkey_to_check = M1*rm.generator(r,m)

	pubkey_to_check *= P1

	print ("[i]\tPubkey, based on found M\' and P\' :\n", pubkey_to_check)

	if pubkey_to_check == gpub:

		print_log("Attack succeeded!", mode='good')
		return True
	else:
		print_log("Attack failed :(", mode='bad')
		return False
	


def print_header():
	print_log ('Welcome to Just!Attack 1.0', mode='info')
	print_log ('Please choose your next action:', mode='info')


def GUI(pubkey = matrix.Matrix(), M1 = matrix.Matrix(), P1 = matrix.Matrix(), key_generated = False, attack_performed = False):
	print_log ('1. (Re)generate keys', mode='info')
	print_log ('2. Perform attack on the generated public key', mode='info')
	print_log ('3. Check attack result', mode='info')
	print_log ('0. Exit', mode='info')

	user_action = input()

	if user_action == '1':
		print_log ('Please enter desired r parameter:', mode='info')
		global r 
		r = int(input())
		print_log ('Please enter desired m parameter:', mode='info')
		global m 
		m = int(input())

		print_log ('Generation of pubkey started...', mode='good')

		pubkey= pubkey_gen()

		print_log ('Generation finished.', mode='good')

		#print_log ('There is your M matrix:', mode='info')

		#print(M)

		print_log ('There is your pubkey:', mode='info')
		print(pubkey)

		#print_log ('There is your permutation matrix:', mode='info')
		#print(P)

		key_generated = True


		GUI(pubkey, key_generated = True, attack_performed = False)

	elif user_action == '2':
		if not key_generated:
			print_log ('To perform attack you first need to generate keys. Would you like to generate keys?', mode='bad')
		else:

			print_log ('Attack performing started...', mode='good')

			M1, P1 = perform_attack(pubkey)

			print_log ('Attack performing finished', mode='good')


			print_log ('There is found M\' matrix:', mode='info')

			print(M1)

			print_log ('There is found permutation matrix P\':', mode='info')
			print(P1)

			print_log ('')

			GUI(pubkey, M1, P1, key_generated = True, attack_performed = True)

	elif user_action == '3':
		if not key_generated:
			print_log ('To check the result you first need to generate keys and perform the attack. Would you like to generate keys?', mode='bad')
		elif not attack_performed:
			print_log ('To check the result you first need to perform the attack. Would you like to perform the attack?', mode='bad')
		else:

			check_attack(pubkey, M1, P1)

			GUI(pubkey, M1, P1, key_generated = True, attack_performed = True)
	else:
		print_log ('Bye!', mode='info')
		exit()
		#return


	GUI(pubkey, M1, P1, key_generated, attack_performed)


print_header()
GUI()