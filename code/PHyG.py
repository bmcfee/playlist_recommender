#!/usr/bin/env python
"""(P)ersonalized (Hy)pergraph (P)laylist model"""

import numpy as np
import scipy.sparse
import scipy.optimize

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator

class PlaylistModel(BaseEstimator):

    def __init__(self, n_factors=8, edge_reg=1.0, bias_reg=1.0, user_reg=1.0, 
                 song_reg=1.0, max_iter=10, edge_init=None, bias_init=None, 
                 user_init=None, song_init=None, params='ebus', n_neg=64, n_jobs=1, 
                 memory=None):
        """Initialize a personalized playlist model

        :parameters:
         - n_factors : int >= 0
            Number of latent factors in the personalized model

         - edge_reg : float > 0
            Variance reg on the edge weights

         - bias_reg : float > 0
            Variance reg on song bias

         - user_reg : float > 0
            Variance reg on user latent factors

         - song_reg : float > 0
            Variance reg on song latent factors

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
        
        self.edge_reg = edge_reg
        self.bias_reg = bias_reg
        self.user_reg = user_reg
        self.song_reg = song_reg

        if user_init is not None:
            self.n_factors = user_init.shape[1]

        if song_init is not None:
            self.n_factors = song_init.shape[1]


    def _fit_users(self, bigrams, iter=None):
        # Solve over all users in parallel
        self.u_[:] = np.asarray(Parallel(n_jobs=self.n_jobs)(delayed(user_optimize)(self.n_neg,
                                                                                    self.H_,
                                                                                    self.w_,
                                                                                    self.user_reg,
                                                                                    self.v_,
                                                                                    self.b_,
                                                                                    y) for y in bigrams))


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
            
          - H : scipy.sparse.csr_matrix (n_songs, n_edges)
        '''

        # Stash the hypergraph
        self.H_ = H

        # Convert playlists to bigrams

        pass

#-- Static methods: things that can parallelize

def sample_noise_items(n_neg, H, edge_dist, b, y_pos):
    '''Sample n_neg items from the noise distribution, forbidding observed samples y_pos.

    '''

    y_forbidden = set(y_pos)

    edge_dist = edge_dist / np.sum(edge_dist)

    full_item_dist = np.exp(b)

    # Knock out forbidden items
    full_item_dist[list(y_forbidden)] = 0.0

    noise_ids = []

    while len(noise_ids) < n_neg:
        # Sample an edge
        edge_id = np.flatnonzero(np.random.multinomial(1, edge_dist))

        item_dist = H[edge_id] * full_item_dist

        item_dist_norm = np.sum(item_dist)

        if not item_dist_norm:
            continue

        item_dist /= item_dist_norm

        while True:
            new_item = np.flatnonzero(np.random.multinomial(1, item_dist))

            if new_item not in y_forbidden:
                break

        y_forbidden.add(new_item)
        noise_ids.append(new_item)
        full_item_dist[new_item] = 0.0

    return noise_ids

def make_bigram_weights(H, s, t, weight):
    if s == -1:
        # This is a phantom state, so we only care about t
        my_weight = H[t].multiply(weight)
    else:
        # Otherwise, (s,t) is a valid transition, so use both
        my_weight = H[s].multiply(H[t]).multiply(weight)

    # Normalize the edge probabilities
    my_weight /= np.sum(my_weight)
    return my_weight

def user_optimize(n_noise, H, w, reg, v, b, bigrams, u0=None):
    '''Optimize a user's latent factor representation

    :parameters:
        - n_noise : int > 0
          # noise items to sample

        - H : scipy.sparse.csr_matrix, shape=(n_songs, n_edges)
          The hypergraph adjacency matrix

        - w : ndarray, shape=(n_edges,)
          edge weight array

        - reg : float >= 0
          Regularization penalty

        - v : ndarray, shape=(n_songs, n_factors)
          Latent factor representation of items

        - b : ndarray, shape=(n_songs,)
          Bias terms for songs

        - bigrams : iterable of tuples (s, t)
          Observed bigrams for the user

        - u0 : None or ndarray, shape=(n_factors,)
          Optional initial value for u
    '''

    # 1. Extract positive ids
    pos_ids = [t for (s, t) in bigrams]

    exp_w = np.exp(w)

    # 2. Sample n_neg songs from the noise model (u=0)
    noise_ids = sample_noise_items(n_noise, H, exp_w, b, pos_ids)

    # 3. Compute and normalize the bigram transition weights
    #   handle the special case of s==-1 here

    bigram_weights = np.asarray([make_bigram_weights(H, s, t, exp_w) for (s, t) in bigrams])

    # 4. Compute the importance weights for noise samples
    noise_weights = [bigram_weights * H[id].T for id in noise_ids]

    # 5. Construct the inputs to the solver
    y = np.ones(len(pos_ids) + len(noise_ids))
    y[len(pos_ids):] = -1

    # The first bunch are positive examples, and get weight=+1
    weights = np.ones_like(y)
    # The remaining examples get noise weights
    weights[len(pos_ids):] = noise_weights

    ids = np.concatenate([pos_ids, noise_ids])

    return user_optimize_objective(reg, v[ids], b[ids], y, weights, u0=u0)


def user_optimize_objective(reg, v, b, y, omega, u0=None):
    '''Optimize a user vector from a sample of positive and noise items

    :parameters:
        - reg : float >= 0
          Regularization penalty

        - v : ndarray, shape=(m, n_factors)
          Latent factor representations for sampled items

        - b : ndarray, shape=(m,)
          Bias terms for items

        - y : ndarray, shape=(m,)
          Sign matrix for items (+1 = positive association, -1 = negative)

        - omega : ndarray, shape=(m,)
          Importance weights for items

        - u0 : None or ndarray, shape=(n_factors,)
          Initial value for the user vector

    :returns:
        - u_opt : ndarray, shape=(n_factors,)
          Optimal user vector
    '''

    def __user_obj(u):
        '''Optimize the user objective function:

        min_u reg * ||u||^2 + sum_i y[i] * omega[i] * log(1 + exp(-y * u'v[i] + b[i]))

        '''

        # Compute the scores
        scores = y * (v.dot(u) + b)
    
        f = reg * 0.5 * np.sum(u**2) + omega.dot(np.logaddexp(0, -scores))
        
        grad = reg * u - v.T.dot(y * omega / (1.0 + np.exp(scores)))
    
        return f, grad

    # Make sure our data is properly shaped
    assert len(v) == len(b)
    assert len(v) == len(y)
    assert len(v) == len(omega)

    if not u0:
        u0 = np.zeros(len(v))

    u_opt, value, diagnostic = scipy.optimize.fmin_l_bfgs_b(__user_obj, u0)

    # Ensure that convergence happened correctly
    assert diagnostic['warnflag'] == 0

    return u_opt