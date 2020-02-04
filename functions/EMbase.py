import warnings
import numpy as np
import time
from cython_func.EMbase_c import expectation, covariance

# sample
# 

class GaussianMixture(object):
    """Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    n_components : int, defaults to 1.
        The number of mixture components.

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        likelihood gain is below this threshold.

    max_iter : int, defaults to 100.
        Maximum number of EM iterations to perform.

    means_init: array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `_init_params` method.

    covariances_init : array-like, shape (n_components, n_features,n_features), optional
        The user-provided initial covariances, defaults to None.
        If it None, covariances are initialized using the `_init_params` method.

    random_state : int or None, optional (default=None)
        Controls the random seed given to the method chosen to initialize the
        parameters (see `_init_params`).

    verbose : int (0 or 1), default to 0.
        Enable verbose output. If 1, then it prints every verbose_interval steps the 
        iteration step number, the log likelihood and the time needed
        for each step. It also return the log likelihood vector.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.


    Attributes
    ----------

    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like, (n_components, n_features, n_features) 
        The covariance of each mixture component.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model).
    """


    def __init__(self, n_components=1, tol=1e-3, max_iter = 100, random_state=None, means_init=None, covariances_init=None, verbose=0, verbose_interval=10):

        self.n_components=n_components
        self.tol = tol
        self.max_iter = max_iter
        self.fitted=False
        self.random_state = random_state
        self.verbose=verbose
        self.verbose_interval = verbose_interval

        self.weights_ = None
        self.means_= means_init
        self.covariances_= covariances_init 
        self.n_iter_ = None
        self.lower_bound_ = None
        self.iter_ = None


    def fit(self,X):
        """Estimate model parameters with the EM algorithm.

        The method fits the model and sets the parameters. The method 
        iterates between E-step and M-step for ``max_iter`` times until 
        the change of log likelihood is less than``tol``, 
        otherwise, a ``ConvergenceWarning`` is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        self
        """

        self.fit_predict(X)
        return self


    def fit_predict(self, X):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model and sets the parameters. The method 
        iterates between E-step and M-step for ``max_iter`` times until 
        the change of log likelihood is less than``tol``, 
        otherwise, a ``ConvergenceWarning`` is raised.

        After fitting, it predicts the most probable label for the
        input data points.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """

        if (self.means_ is None) or (self.covariances_ is None):
            self._init_params(X)
        self.weights_ = np.ones(self.n_components)/self.n_components

        log_likelihood = -np.infty 

        for n_iter in range(1, self.max_iter+1):

            prev_log_likelihood = log_likelihood
            resp, log_likelihood = self._e_step(X)
            self._m_step(X, resp)

            if self.verbose>0 and n_iter%self.verbose_interval==0:
                print("Log Likelihood : {} ({}/{})".format(log_likelihood, n_iter,self.max_iter))

            if abs(log_likelihood-prev_log_likelihood)<self.tol:
                self.n_iter_ = n_iter
                self.lower_bound_ = log_likelihood
                if self.verbose>0:
                    print("Convergence in {} iterations".format(n_iter))
                break

        if n_iter==self.max_iter:
            warnings.warn('Model did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data.')
        self.fitted = True
        self.iter_ = n_iter

        return resp.argmax(axis=1)


    def predict_proba(self, X):
        """Predict posterior probability of each component given the data.
        Return an error if the model is not fitted upon the call (fitted=Talse)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """

        # WHY NO WEIGHTS !!!!!
        if not self.fitted:
            print("Classifier is not fitted, please call fit first")
            return
        else:
            #resp = np.zeros((X.shape[0], self.n_components))  
            #for idx in range(self.n_components):  
            #    resp[:,idx] = multivariate_normal(mean=self.means_[idx],cov=self.covariances_[idx]).pdf(X)*self.weights_[idx]
            #resp = ( resp.T / resp.T.sum(axis=0) ).T
            return expectation(X,self.means_,self.covariances_, self.weights_, self.fitted)


    def predict(self,X):
        """Predict the labels for the data samples in X using trained model.
        Return an error if the model is not fitted upon the call (fitted=Talse)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """

        if not self.fitted:
            print("Classifier is not fitted, please call fit first")
            return
        else:
            return self.predict_proba(X).argmax(axis=1)


    def _m_step(self, X, resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
            [[[Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.]]] -> Is this really that ????? #####
        """

        self.weights_ = resp.mean(axis=0)
        self.means_ = np.array([np.average(X, axis=0, weights=resp[:,i]) for i in range(resp.shape[1]) ])
        self.covariances_ = np.array([covariance(X, resp[:,i], self.means_[i]) for i in range(resp.shape[1]) ])
        

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_responsibility : array, shape (n_samples, n_components)
            [[[Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.]]] -> Is it really that ??? #######
        """

        return expectation(X,self.means_,self.covariances_, self.weights_, self.fitted)


    def _init_params(self,X):
        """Initialization of the Gaussian mixture parameters if means_ or covariances_ is None.

        The n_components covariances matrices are initialized to identity matrices.
        The n_components mean vectors are filled with random value drawn from uniform distributions that take
        value between the minimum value and the maximum value along the given dimension. 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """

        np.random.seed(self.random_state)
        if self.means_ is None:
            self.means_ = np.hstack([np.random.uniform(np.min(X[:,i]),np.max(X[:,i]),size=(self.n_components,1)) for i in range(X.shape[1])])   
        if self.covariances_ is None:
            self.covariances_ = np.array([np.eye(X.shape[1]) for i in range(self.n_components)])














