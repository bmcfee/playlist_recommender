#!/usr/bin/env python
"""(P)ersonalized (Hy)pergraph (P)laylist model"""

import numpy as np
import scipy.sparse
import scipy.optimize

from sklearn.base import BaseEstimator

class PlaylistModel(BaseEstimator):

    def __init__(self, 
                 n_factors=16, 
                 edge_prior=1.0, 
                 bias_prior=1.0, 
                 user_prior=1.0,
                 song_prior=1.0,
                 max_iter=10,
                 edge_init=None,
                 bias_init=None,
                 user_init=None,
                 song_init=None,
                 params='ebus'):
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
        """

        self.max_iter   = max_iter
        self.params     = params
        self.n_factors  = n_factors
        
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

    def _fit_edges(self, H, P):
        # // pre-compute the user-edge affinity matrix
        # Q = 1.0/(H' * exp(u.dot(V.T) + b))
        #
        # Lower-bounding objective function:
        # f         = 0
        # grad_f    = 0
        #
        #
        # for i in users
        #
        #   for p in playlists[i]
        #       // Hack the t=-1 case here
        #
        #       t       = p[0]
        #       hq      = H[t,:] * q[i]
        #       hq      = hq / sum(hq)
        #       lw      = log_sum_exp(w)
        #       f       += w' * hq - lw
        #       grad_f  += hq - soft_max(w)
        #
        #       for (prev, cur) in zip(p[:-1], p[1:])
        #           // compute coincident edges
        #           hq = (H[prev, :] & H[cur, :]) * q[i]
        #           hq = hq / sum(hq)
        #
        #           // compute edge-selector normalization
        #           // better to do this robustly, log_sum_exp(w[H[prev]])
        #           lw = log_sum_exp(w[ H[prev] ])
        #           
        #           f       +=  w' * hq - lw
        #           grad_f  +=  hq - softmax(w[H[prev]])
        # 
        # f         += -0.5 * self.bias_prior * (w' * w)
        # grad_f    += - self.bias_prior * w

        pass

    def fit(self, H, P, params=None):
        """Fit the playlist model.

        :parameters:
        - H : scipy.sparse, shape=(n_songs, n_edges), dtype=bool
            The song-edge coincidence matrix

        - P : dict of list of list
            `P[i]` is a list of playlists for user `i`.
            `P[i][j]` is the `j`th playlist of user `i`.
            `P[i][j][k]` is the id of the `k`th song.

            Keys of P need to be integers.

        - params : None or str
            If `str`, fit only the specified parameters.
            If `None`, fit all parameters set at initialization.
        """

        # First, make sure we have all our ducks in a row
        n_songs, n_edges = H.shape[0]
        n_users = len(P)

        if self.w_ is None:
            self.w_ = np.zeros(n_edges, dtype=np.float32)

        # Probably it would be smarter to call out to an implicit-feedback
        # collaborative filter model for initialization
        if self.b_ is None:
            self.b_ = np.zeros(n_songs, dtype=np.float32)

        if self.u_ is None:
            self.u_ = np.random.randn(n_users, self.n_factors, dtype=np.float32)

        if self.v_ is None:
            self.v_ = np.zeros((n_songs, self.n_factors), dtype=np.float32)

        if params is not None:
            params = self.params

        for _ in range(self.max_iter):
            if 'e' in params:
                # Fit the edge weights
                self._fit_edges(H, P)

            if 'b' in params:
                # Fit the song biases
                pass
            if 'u' in params:
                # Fit the user factors
                pass
            if 's' in params:
                # Fit the song factors
                pass
                
            # If we're only fitting one parameter, we can break
            if len(params) <= 1:
                break
        return self
