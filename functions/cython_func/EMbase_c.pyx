#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from cython.parallel cimport prange
import numpy
cimport numpy
from libc.math cimport exp as cexp

DEF LOG_2_PI = 0.79817986835


cdef double log_probability(double* X, double* mu, double* inv_cov, double log_det, int d) nogil:
    cdef int i, j
    cdef double logp = 0.0
    
    for i in range(d):
        for j in range(d):
            logp += (X[i] - mu[i]) * (X[j] - mu[j]) * inv_cov[i + j*d]
    
    return -0.5 * (d * LOG_2_PI + log_det + logp)
      

cpdef numpy.ndarray _expectation( numpy.ndarray X_ndarray, numpy.ndarray mu_ndarray, numpy.ndarray cov_ndarray ):
    cdef int i, n = X_ndarray.shape[0], d = X_ndarray.shape[1]
    cdef double log_det = numpy.linalg.slogdet(cov_ndarray)[1]
    cdef numpy.ndarray inv_cov_ndarray = numpy.linalg.inv(cov_ndarray)
    cdef numpy.ndarray r_ndarray = numpy.zeros(n)
    
    cdef double* inv_cov = <double*> inv_cov_ndarray.data
    cdef double* mu = <double*> mu_ndarray.data
    cdef double* X = <double*> X_ndarray.data
    cdef double* r = <double*> r_ndarray.data
    
    for i in prange(n, nogil=True, schedule='guided'):
        r[i] = cexp(log_probability(X + i*d, mu, inv_cov, log_det, d ))
    
    return r_ndarray


def expectation(X, mu, cov, weights, fitted):

    resp = numpy.hstack([_expectation(X, m, c + 1e-10*numpy.eye(X.shape[1]))[:, numpy.newaxis] for m, c in zip(mu, cov)]) # n * n_comp
    resp = resp * weights # weights =  n_comp, 

    if not fitted:
        log_likelihood = numpy.sum(numpy.log(numpy.sum(resp,axis=1)))/X.shape[0]

    resp = ( resp.T / resp.T.sum(axis=0) ).T

    if not fitted:
        return resp, log_likelihood
    else:
        return resp


cpdef numpy.ndarray covariance(numpy.ndarray X, numpy.ndarray weights, numpy.ndarray mu):
    cdef int i, j, n = X.shape[0], d = X.shape[1]
    cdef numpy.ndarray cov = numpy.zeros((d, d))
    cdef double w_sum
    
    for i in range(d):
        for j in range(i+1):
            cov[i, j] = weights.dot( (X[:,i] - mu[i])*(X[:,j] - mu[j])  )
            cov[j, i] = cov[i, j]
    
    w_sum = weights.sum()
    return cov / w_sum