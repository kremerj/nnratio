""" nnratio - Nearest neighbor density ratio estimator
    Copyright (C) 2015 Jan Kremer <jan.kremer@di.ku.dk>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import KFold
from joblib import Parallel, delayed

def loss(w_tr, w_te):
    """ Model loss used to evaluate the estimated weights
        w_tr: estimated weights on training set
        w_te: estimated weights on test set

        returns: model loss over the supplied training and test samples
    """
    return (np.mean(w_tr.T**2.0,axis=0) - 2.0 * np.mean(w_te.T,axis=0)).T

def cv_loss(X_tr, X_te, kf_tr, kf_te, K, n_cv=5):
    """ Cross validtation loss helper function over given folds
        X_tr: training sample
        X_te: test sample
        kf_tr: K-Folds over the training sample
        kf_te: K-Folds over the test sample
        K: K neighbors to consider
        n_cv: number of folds used for cross-validation

        returns: cross-validation loss
    """ 
    tr_iter = iter(kf_tr)
    te_iter = iter(kf_te)
    
    cv_losses = np.zeros(n_cv)
    
    for i in range(n_cv):
        tr_ind, tr_val_ind = tr_iter.next()
        te_ind, te_val_ind = te_iter.next()
        
        tr     = X_tr[tr_ind,:]
        tr_val = X_tr[tr_val_ind,:] 
        te     = X_te[te_ind,:]
        te_val = X_te[te_val_ind,:]
        
        knn_model = NearestNeighborsRatioEstimator(n_neighbors=K)
        knn_model.fit(tr, te)
        
        w_tr   = knn_model.compute_weights(tr_val)
        w_te   = knn_model.compute_weights(te_val)
        
        cv_losses[i] = loss(w_tr,w_te)
        
    val_loss = np.mean(cv_losses)
    return val_loss

class NearestNeighborsRatioEstimator(object):
    """ Nearest neighbor ratio estimator
    """
    def __init__(self, n_neighbors=2):
        """ Instantiates the learner.
            n_neighbors: number of neighbors for KNN estimator
        """
        
        self.n_neighbors = n_neighbors

    def get_params(self, deep=True):
        """ Get parameters (for scikit-learn)
        """
        return {"n_neighbors": self.n_neighbors}

    def set_params(self, **parameters):
        """ Set parameters (for scikit-learn)
        """
        for parameter, value in parameters.items():
            self.setattr(parameter, value)

    def fit(self, X_tr, X_te):
        """ Fit the model
            X_tr: training sample
            X_te: test sample
        """
        self.n_tr = X_tr.shape[0]
        self.n_te = X_te.shape[0]

        # build kd-trees for both domains
        self.nbrs_tr = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='kd_tree').fit(X_tr)
        self.nbrs_te = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='kd_tree').fit(X_te)

    def fit_cv(self, X_tr, X_te, K_list, n_cv=5, n_jobs=0):
        """ Fit the model using cross-validation to find an optimal number of K neighbors
            X_tr: training sample
            X_te: test sample
            K_list: list of K values to consider for cross-validation
            n_cv: number of folds for cross-validation
            n_jobs: number of jobs to use when running cross-validation in parallel, if 0, take len(K_list) jobs
        """
        if n_jobs == 0:
            n_jobs = len(K_list)
        n_train = X_tr.shape[0]
        n_test  = X_te.shape[0]
        kf_tr   = KFold(n_train, n_folds=n_cv)
        kf_te   = KFold(n_test,  n_folds=n_cv)
        self.losses = Parallel(n_jobs=n_jobs)(delayed(cv_loss)(X_tr,X_te,kf_tr,kf_te,K,n_cv) for K in K_list)
        self.n_neighbors = K_list[np.argmin(self.losses)]
        self.fit(X_tr,X_te)

    def compute_weights(self, X_ev):
        """ Predicts weights for a set of (evaluation) patterns.
            X_ev: sample of data points at which we evaluate the density ratio weights

            returns: weights for each data point in the given sample
        """
        # get K nearest neighbors (and radii) from training domain
        distances, ind = self.nbrs_tr.kneighbors(X_ev)
        radii = distances[:,-1] 
        
        # compute weights
        weights = np.zeros(X_ev.shape[0])
        for i in xrange(X_ev.shape[0]):
            # count number of numerator sample within the current radius of the query sample
            weights[i] = len(self.nbrs_te.radius_neighbors(X_ev[i,:].reshape(1, -1), radius=radii[i], return_distance=False)[0])
        # divide K denominator samples and normalize by the ratio of denominator to numerator samples
        weights *= float(self.n_tr) / float(self.n_neighbors * self.n_te)
                      
        return weights
