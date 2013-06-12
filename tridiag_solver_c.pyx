from __future__ import division
#Tridiagonal solver
cimport numpy as np
import cython

#cython: boundscheck=False
#cython: wraparound=False


# Standard triadiagonal solver
# see http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm for details
# Implementation uses numpy arrays in combination with Cython definitions
## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
@cython.profile(False)
@cython.boundscheck(False)
def SolveTridiagonal(np.ndarray a not None,np.ndarray b not None,np.ndarray c not None,np.ndarray d not None):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    :a subdiagonal with initial 0
    :b diagonal
    :c super diagonal with trailing 0
    :d value vector we are solving for
    '''
    cdef int nf = len(a)     # number of equations
    cdef np.ndarray ac = a.copy()
    cdef np.ndarray bc = b.copy()
    cdef np.ndarray cc = c.copy()
    cdef np.ndarray dc = d.copy()
     
    cdef int it
    for it in xrange(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]

    cdef np.ndarray xc = ac
    xc[-1] = dc[-1]/bc[-1]
	
    cdef int il
    for il in xrange(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
 
    del ac, bc, cc, dc  # delete variables from memory
 
    return xc

