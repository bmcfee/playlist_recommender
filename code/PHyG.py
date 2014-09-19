#!/usr/bin/env python
"""(P)ersonalized (Hy)pergraph (P)laylist model"""

import numpy as np
import scipy.sparse
import scipy.optimize
import scipy.misc

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator

__EXP_BOUND = 80.0

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
        self.u_[:] = Parallel(n_jobs=self.n_jobs)(delayed(user_optimize)(self.n_neg,
                                                                         self.H_, self.w_,
                                                                         self.user_reg, self.v_,
                                                                         self.b_, y) for y in bigrams)


    def _fit_songs(self, iter=None):

        pass


    def _fit_bias(self, iter=None):

        pass


    def _fit_edges(self, bigrams, w0=None, iter=None):
        '''Update the edge weights'''

        # The edge weights
        Z = 0

        # num_usage[s] counts bigrams of the form (s, .)
        num_usage = 0

        # num_playlists counts all playlists
        num_playlists = 0

        # Scatter-gather the bigram statistics over all users
        for Z_i, nu_i, np_i in Parallel(n_jobs=self.n_jobs)(delayed(edge_user_weights)(self.H_, 
                                                                          self.H_T_, 
                                                                          self.u_, 
                                                                          idx, 
                                                                          self.v_, 
                                                                          self.b_, y) for (idx, y) in enumerate(bigrams)):
            Z += Z_i
            num_usage += nu_i
            num_playlists += np_i

        Z = np.asarray(Z.todense()).ravel()
        num_usage = np.asarray(num_usage.todense()).ravel()

        def __edge_objective(w):
            
            obj = self.edge_reg * 0.5 * np.sum(w**2)
            grad = self.edge_reg * w

            obj += - Z.dot(w)
            grad += -Z

            lse_w = scipy.misc.logsumexp(w)
            exp_w = np.exp(w)
            Hexpw = self.H_.multiply(exp_w)

            # Compute stable item-wise log-sum-exp slices
            Hexpw_norm = np.empty_like(num_usage)
            Hexpw_norm[:] = [scipy.misc.logsumexp(np.take(w, hid.indices)) for hid in self.H_]

            obj += num_usage.dot(Hexpw_norm) + num_playlists * lse_w

            grad += np.ravel( (num_usage * np.exp(-Hexpw_norm)).dot(Hexpw))
            grad += exp_w * (num_playlists * np.exp(-lse_w))

            return obj, grad


        if not w0:
            w0 = np.zeros(self.H_.shape[0])

        bounds = [(-__EXP_BOUND, __EXP_BOUND)] * len(w0)
        w_opt, value, diagnostic = scipy.optimize.fmin_l_bfgs_b(__edge_objective, 
                                                                w0, 
                                                                bounds=bounds)

        # Ensure that convergence happened correctly
        assert diagnostic['warnflag'] == 0

        self.w_ = w_opt


    def fit(self, playlists, H):
        '''fit the model.

        :parameters:
          - playlists : dict (users => list of playlists)
            
          - H : scipy.sparse.csr_matrix (n_songs, n_edges)
        '''

        # Stash the hypergraph and its transpose
        self.H_ = H.tocsr()
        self.H_T_ = H.T.tocsr()

        # Convert playlists to bigrams
        #   bigrams[user_id] = [(s,t) for all pairs (s,t) in playlists[user_id]]

        pass

#-- Static methods: things that can parallelize

#--- common functions to user, item, and bias optimization:

def make_bigram_weights(H, s, t, weight):
    if s is None:
        # This is a phantom state, so we only care about t
        my_weight = H[t].multiply(weight)
    else:
        # Otherwise, (s,t) is a valid transition, so use both
        my_weight = H[s].multiply(H[t]).multiply(weight)

    # Normalize the edge probabilities
    my_weight /= my_weight.sum()
    return my_weight

def sample_noise_items(n_neg, H, edge_dist, b, y_pos):
    '''Sample n_neg items from the noise distribution, forbidding observed samples y_pos.

    '''

    y_forbidden = set(y_pos)

    edge_dist = np.asarray(edge_dist / edge_dist.sum())

    full_item_dist = np.exp(b)

    # Knock out forbidden items
    full_item_dist[list(y_forbidden)] = 0.0

    noise_ids = []

    while len(noise_ids) < n_neg:
        # Sample an edge
        edge_id = np.flatnonzero(np.random.multinomial(1, edge_dist))[0]

        item_dist = np.ravel(H[:, edge_id].T.multiply(full_item_dist))

        item_dist_norm = np.sum(item_dist)

        if not item_dist_norm:
            continue

        item_dist /= item_dist_norm

        while True:
            new_item = np.flatnonzero(np.random.multinomial(1, item_dist))[0]

            if new_item not in y_forbidden:
                break

        y_forbidden.add(new_item)
        noise_ids.append(new_item)
        full_item_dist[new_item] = 0.0

    return noise_ids

def generate_user_instance(n_neg, H, edge_dist, b, bigrams):
    '''Generate a subproblem instance.

    Inputs:
        - n_neg : # of negative samples
        - H : hypergraph incidence matrix
        - edge_dist : weights of the edges of H
        - b : bias factors for items
        - bigrams : list of tuples (s, t) for the user

    Outputs:
        - y : +-1 label vector
        - weights : importance weights, shape=y.shape
        - ids : list of indices for the sampled points, shape=y.shape
    '''

    # 1. Extract positive ids
    pos_ids = [t for (s, t) in bigrams]

    exp_w = scipy.sparse.lil_matrix(np.exp(edge_dist))

    # 2. Sample n_neg songs from the noise model (u=0)
    noise_ids = sample_noise_items(n_neg, H, np.ravel(exp_w.todense()), b, pos_ids)

    # 3. Compute and normalize the bigram transition weights
    #   handle the special case of s==None here

    bigram_weights = np.asarray([make_bigram_weights(H, s, t, exp_w) for (s, t) in bigrams])

    # 4. Compute the importance weights for noise samples
    noise_weights = np.sum(np.asarray([[(H[id] * bg.T).todense() for id in noise_ids] for bg in bigram_weights]), axis=0).ravel()

    # 5. Construct the inputs to the solver
    y = np.ones(len(pos_ids) + len(noise_ids))
    y[len(pos_ids):] = -1

    # The first bunch are positive examples, and get weight=+1
    weights = np.ones_like(y)

    # The remaining examples get noise weights
    weights[len(pos_ids):] = noise_weights

    ids = np.concatenate([pos_ids, noise_ids])

    return y, weights, ids

#--- edge optimization

def edge_user_weights(H, H_T, u, idx, v, b, bigrams):
    '''Compute the edge weights and transition statistics for a user.'''

    # First, compute the user-item affinities
    item_scores = v.dot(u[idx]) + b

    # Now aggregate by edge
    edge_scores = (H_T * item_scores)**(-1.0)

    # num playlists is the number of bigrams where s == None
    num_playlists   = 0
    num_usage       = np.zeros(len(b))
    Z = 0

    # Now sum over bigrams
    for s, t in bigrams:
        Z = Z + make_bigram_weights(H, s, t, edge_scores)
        if s is None:
            num_playlists += 1
        else:
            num_usage[s] += 1

    return Z, num_usage, num_playlists

#--- user optimization
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

    y, weights, ids = generate_user_instance(n_noise, H, w, b, bigrams)

    return user_optimize_objective(reg, np.take(v, ids, axis=0), np.take(b, ids), y, weights, u0=u0)

# TODO:   2014-09-18 10:46:30 by Brian McFee <brian.mcfee@nyu.edu>
#  refactor this to support item and bias optimization
#   include an additional parameter for the regularization term/lagrangian matrix
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
    
        f = reg * 0.5 * np.sum(u**2) 
        f += omega.dot(np.logaddexp(0, -scores))
        
        grad = reg * u 
        grad -= v.T.dot(y * omega / (1.0 + np.exp(scores)))
    
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
