#Tridiagonal solver
import numpy as np
from numpy import random as rnd
from tridiag_solver_c import SolveTridiagonal as SolveTridiagonalC
import time


# Standard triadiagonal solver
# see http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm for details
# implementation uses numpy arrays instead
## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def SolveTridiagonal(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    :a subdiagonal with initial 0
    :b diagonal
    :c super diagonal with trailing 0
    :d value vector we are solving for
    '''
    nf = len(a)     # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
    for it in xrange(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
 
	xc = ac.copy()
	xc[-1] = dc[-1]/bc[-1]
 
    for il in xrange(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
 
    del ac, bc, cc, dc  # delete variables from memory
 
    return xc

# NB convergence is not very good so dont use for FD methods
def PSORTridiagonal(a,b,c,d):
	'''
		Implemetation of projected successive over relaxation
		for tridiagonal matrix only.
		 
		:a subdiagonal with initial 0
		:b diagonal
		:c super diagonal with trailing 0
		:d value vector we are solving for
	'''
	
	# Tolerance
	eps = 0.00000001
	max_steps = 50
	n = len(d)
	
	assert n > 2
	
	#This is the optimal omega for Finite Difference Method shaped matrix
	#Look up Yang and Gobbert: "The Optimal Relaxation Parameter for the SOR Method Applied to a Classical Model Problem"
	#omega = 2.0/(1.0 + np.sin(np.pi/(n-1)))
	#print n," ", np.sin(np.pi*(1.0/(n-1)))
	omega = 1.35

	V = np.zeros(n)
	Vold = np.ones(n)

	step = 0
	while sum((V - Vold)**2) > eps and step < max_steps:
		Vold = V.copy()
		V[0] = (1-omega)*V[0] + (omega/b[0])*(d[0] - c[0]*V[1])
		for j in xrange(1,n-1):
			V[j] = (1-omega)*V[j] + (omega/b[j])*(d[j] - a[j]*V[j-1]- c[j]*V[j+1])
			
		V[n-1] = (1-omega)*V[n-1] + (omega/b[n-1])*(d[n-1] - a[n-1]*V[n-2])
		step += 1

	return V

def RunTest():
	""" 
		Test if the above implementations for tridiagonal solvers
		match the ones natively provided by numpy
			
		no params
	"""
	
	tolerance = 0.00000000000001
	toleranceForItr = 0.00001 #Tolerance for iterative procedures
	mat = np.array([[3.0, 1.0, 0.0],
					[1.11, 3.0, 1.2],
					[0.0, 0.2, 3.0]])
					
	Y = np.array([1.0,2.0,3.0])

	supDiag = mat.diagonal(1)
	supDiag = np.append(supDiag,[0.0])
	subDiag = mat.diagonal(-1)
	subDiag = np.append([0.0],subDiag)
	Diag = mat.diagonal()

	sol1 = np.linalg.solve(mat,Y)
	sol2 = SolveTridiagonal(subDiag,Diag,supDiag,Y)
	sol3 = PSORTridiagonal(subDiag,Diag,supDiag,Y)
	sol4 = SolveTridiagonalC(subDiag,Diag,supDiag,Y)
	
	print "Test solutions (should be all equal(ish))"
	print "Numpy: ", sol1
	print "Thomson: ", sol2
	print "PSOR: ", sol3
	print "Thomson2: ", sol4
	
	#print "Mostly equal: ", max(abs(sol1-sol2)) < tolerance and max(abs(sol1-sol3)) < toleranceForItr
	
def RunSpeedTest():
	'''
	Compare speed of Thomson algoritm vs PSOR vs Numpy solver
	
	'''
	n = 1000
	n_tests = 100
	
	print "Generating random inputs"
	matA = rnd.randn(n_tests,n)*10
	matB = rnd.randn(n_tests,n)*10
	matC = rnd.randn(n_tests,n)*10
	matD = rnd.randn(n_tests,n)*10
	
	sol1 = np.zeros((n_tests,n))
	sol2 = np.zeros((n_tests,n))
	
	print "Starting PSOR tests"
	t1 = time.time()
	for k in range(n_tests):
		sol1[k] = PSORTridiagonal(matA[k],matB[k],matC[k],matD[k])
	t2 = time.time()
	print "Finished PSOR tests in ",(t2 - t1),"s"
	
	t1 = time.time()
	print "Starting Thomson algorithm test"
	for k in range(n_tests):
		sol2[k] =SolveTridiagonal(matA[k],matB[k],matC[k],matD[k])
	t2 = time.time()
	print "Finished thomson in ",(t2 - t1),"s"
	
	
RunTest()
#RunSpeedTest()


