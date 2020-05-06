
import random

from blincodes.codes import rm
from blincodes import matrix
from blincodes import vector

r = 3
m = 5

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

def gauss_codewords_with_weight_in_range (M, eps):

	max_tries = 500
	desired_weight_min = 2**(m - r) - 1

	desired_weight_max = 2**(m - 2*r + 1)*(2**r - 1)*eps - 1

	codewords = ()

	for i in range(0, max_tries):
		print ("[ ]\ttry to find codewords", i)
		sample = random.sample(range(0, n), k)

		#ssample = sorted(sample)
		#print(ssample, "Len:", len(ssample))

		rmcg = rmc.gaussian_elimination(sample)

		rmcgr = rmcg.submatrix(sample, True)

		
		j = 0
		for row in rmcgr:
			
			if row.hamming_weight >= desired_weight_min and row.hamming_weight <= desired_weight_max:

				codewords.add(rmcg[j])

				if len(codewords) == M:
					return codewords

			j+=1
	print("[-]\tout of tries while searching for codewords")


rmcgac = gauss_by_min_weighted_row()

print("Result:")
print(rmcgac)
