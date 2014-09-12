#!/usr/bin/env python
"""(P)ersonalized (Hy)pergraph (P)laylist model"""

import numpy as np
import scipy.sparse
import scipy.optimize
import joblib

from sklearn.utils.extmath import logsumexp
from sklearn.base import BaseEstimator

class PlaylistModel(BaseEstimator):

    def __init__(self, n_factors=8, edge_prior=1.0, bias_prior=1.0, user_prior=1.0, 
                 song_prior=1.0, max_iter=10, edge_init=None, bias_init=None, 
                 user_init=None, song_init=None, params='ebus', n_neg=64, n_jobs=1, 
                 memory=None):
        """Initialize a personalized playlist model

        :parameters:
         - n_factors : int >= 0
            Number of latent factors in the personalized model

         - edge_prior : float > 0
            Variance prior on the edge weights

         - bias_prior : float > 0
            Variance prior on song bias

         - user_prior : float > 0
            Variance prior on user latent factors

         - song_prior : float > 0
            Variance prior on song latent factors

         - max_iter : int > 0
            Number of optimization steps

         - edge_init : ndarray or None
            Initial value of edge weight vector

         - bias_init : ndarray or None
            Initial value of song bias vector

         - user_init : None or ndarray, shape=(n_users, n_factors)
            Initial value of user latent factor matrix

         - song_init : None or ndarray, shape=(n_songs, n_factors)
            Initial value of song latent factor matrix

         - params : str
            Which parameters to fit:
            - 'e' : edge weights
            - 'b' : song bias
            - 'u' : users
            - 's' : songs

         - n_neg : int > 0
            Number of negative examples to draw for each user sub-problem.

         - n_jobs : int
            Maximum number of jobs to run in parallel


         - memory : None or joblib.Memory
            optional memory cache object

        """

        if 'u' not in params or 'v' not in params:
            n_factors = 0

        self.max_iter   = max_iter
        self.params     = params
        self.n_factors  = n_factors
        self.n_neg      = n_neg
        self.n_jobs     = n_jobs

        self.w_         = edge_init
        self.b_         = bias_init
        self.u_         = user_init
        self.v_         = song_init
        
        self.edge_prior = edge_prior
        self.bias_prior = bias_prior
        self.user_prior = user_prior
        self.song_prior = song_prior

        if user_init is not None:
            self.n_factors = user_init.shape[1]

        if song_init is not None:
            self.n_factors = song_init.shape[1]

        if memory is not None:
            self.sample_negatives = memory.cache(self.sample_negatives)


    def sample_negatives(self, user_id, H, iter):
        '''generate n_neg negative samples for the user'''

        pass

    def _fit_users(self, iter=None):
        # Generate negative samples
        # Solve over all users in parallel
        
        # TODO:   2014-09-12 14:14:50 by Brian McFee <brian.mcfee@nyu.edu>
        #   Probably the best way to do this is to memmap the V and H matrices
        #   It might even be worth memmap/preallocate the U matrix as an output space, and just
        #   send the column number as input

        # Samples might be best generated by passing references to the mem-cached negative sample generator

        pass

    def _fit_songs(self, iter=None):

        pass

    def _fit_bias(self, iter=None):

        pass

    def _fit_edges(self, iter=None):

        pass

    def fit(self, playlists, H):
        '''fit the model.

        :parameters:
          - playlists : dict (users => list of playlists)
            
          - H : sparse matrix (n_songs, n_edges)
        '''
        pass
