#!/usr/bin/env python
"""(P)ersonalized (Hy)pergraph (P)laylist model"""

import numpy as np
import scipy.sparse
import scipy.optimize
import scipy.misc

import logging
import time

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator

_EXP_BOUND = 20.0

L = logging.getLogger(__name__)


class PlaylistModel(BaseEstimator):
    '''Personalized hypergraph random walk playlist model'''

    def __init__(self, n_factors=8, edge_reg=1.0, bias_reg=1.0, user_reg=1.0,
                 song_reg=1.0, max_iter=10, edge_init=None, bias_init=None,
                 user_init=None, song_init=None, params='ebus', n_neg=64,
                 max_admm_iter=50,
                 n_jobs=1,
                 verbose=0,
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

         - edge_init : ndarray shape=(n_features,) or None
            Initial value of edge weight vector

         - bias_init : ndarray shape=(n_songs,) or None
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

         - max_admm_iter : int > 0
            Maximum number of ADMM iterations to run in the inner loop.
            Set to np.inf for full convergence tests.

         - n_jobs : int
            Maximum number of jobs to run in parallel

         - verbose : int >= 0
            Verbosity (logging) level

         - memory : None or joblib.Memory
            optional memory cache object

        """

        # If we don't learn latent factors,
        # set n_factors to 1 and pin the user variables to 0
        if 'u' not in params and 's' not in params:
            n_factors = 1

        self.max_iter = max_iter
        self.max_admm_iter = max_admm_iter
        self.n_jobs = n_jobs

        self.params = params
        self.n_factors = n_factors
        self.n_neg = n_neg

        self.w_ = edge_init
        self.b_ = bias_init
        self.u_ = user_init
        self.v_ = song_init

        self.edge_reg = edge_reg
        self.bias_reg = bias_reg
        self.user_reg = user_reg
        self.song_reg = song_reg

        self.verbose = verbose

        L.setLevel(self.verbose)

        if user_init is not None:
            self.n_factors = user_init.shape[1]

        if song_init is not None:
            self.n_factors = song_init.shape[1]

    def _fit_users(self, bigrams):
        # Solve over all users in parallel
        tic = time.time()
        self.u_[:] = Parallel(n_jobs=self.n_jobs)(delayed(user_problem)(self.n_neg,
                                                                        self.H_,
                                                                        self.w_,
                                                                        self.user_reg,
                                                                        self.v_,
                                                                        self.b_,
                                                                        y)
                                                  for y in bigrams)
        toc = time.time()
        L.debug('  [USER] Fit %d user factors in %.3f seconds',
                len(self.u_),
                toc - tic)

    def _fit_songs(self, bigrams):

        # 1. generate noise instances:
        #   for each user subproblem, we need:
        #       y[i]        # pos/neg labels for each of ids_i
        #       weights[i]  # importance weights for each of ids_i
        #       ids[i]      # indices of items for this subproblem
        #

        n_songs = self.H_.shape[0]

        tic = time.time()
        subproblems = Parallel(n_jobs=self.n_jobs)(delayed(generate_user_instance)(self.n_neg,
                                                                                   self.H_,
                                                                                   self.w_,
                                                                                   y,
                                                                                   b=self.b_)
                                                   for y in bigrams)

        # 2. collect usage statistics over subproblems
        #       Z[i] = count subproblems using item i
        #

        counts = np.zeros(n_songs)
        duals = []
        Ai = []

        V = self.v_.copy()

        for sp in subproblems:
            ids_i = sp[-1]
            counts[ids_i] += 1.0
            duals.append(np.zeros((len(ids_i), self.n_factors)))
            Ai.append(np.take(V, ids_i, axis=0))

        toc = time.time()
        L.debug('  [SONG] Initialized %d subproblems in %.3f seconds',
                len(subproblems),
                toc - tic)

        # 3. initialize ADMM parameters
        #   a. V <- self.v_
        #   b. Lambda[i] = np.zeros( (len(ids_i), self.n_factors) )
        #   c. rho = rho_init
        #

        rho = 1.0

        # 4. ADMM loop
        #   a. [parallel] solve each user problem:
        #           (i, S[i], y[i], weights[i], Lambda[i], rho, U, V, b)
        #
        #           equivalently, pass ids, so that
        #               S*V == V[ids] == np.take(V, ids, axis=0)
        #           this buys about a 4x speedup
        #
        #           (i, y[i], weights[i], ids, Lambda[i], rho, U, V, b)
        #

        for step in range(self.max_admm_iter):
            tic = time.time()
            Ai = Parallel(n_jobs=self.n_jobs)(delayed(item_factor_optimize)(i,
                                                                            subproblems[i][0],
                                                                            subproblems[i][1],
                                                                            subproblems[i][2],
                                                                            duals[i],
                                                                            rho,
                                                                            self.u_,
                                                                            V,
                                                                            self.b_,
                                                                            Aout=Ai[i])
                                              for i in range(len(subproblems)))
            toc = time.time()
            L.debug('  [SONG] [%3d/%3d] Solved %d subproblems in %.3f seconds',
                    step,
                    self.max_admm_iter,
                    len(subproblems),
                    toc - tic)

            # Kill the old V
            tic = time.time()
            V.fill(0.0)
            for sp_i, a_i, d_i in zip(subproblems, Ai, duals):
                ids_i = sp_i[-1]
                V[ids_i, :] += (a_i + d_i)

            # Compute the normalization factor
            my_norm = np.reshape(1.0/(counts + self.song_reg / rho), ((-1, 1)))

            # Broadcast the normalization
            V[:] *= my_norm

            toc = time.time()
            L.debug('  [SONG] [%3d/%3d] Gathered solutions in %.3f seconds',
                    step,
                    self.max_admm_iter,
                    toc - tic)

            # Update the residual*
            tic = time.time()
            for sp_i, a_i, d_i in zip(subproblems, Ai, duals):
                ids_i = sp_i[-1]
                d_i[:] = d_i + a_i - np.take(V, ids_i, axis=0)
            toc = time.time()
            L.debug('  [SONG] [%3d/%3d] Updated %d residuals in %.3f seconds',
                    step,
                    self.max_admm_iter,
                    len(subproblems),
                    toc - tic)

        self.v_[:] = V

    def _fit_bias(self, bigrams):

        n_songs = self.H_.shape[0]

        # Generate the sub-problem instances
        tic = time.time()
        subproblems = Parallel(n_jobs=self.n_jobs)(delayed(generate_user_instance)(self.n_neg,
                                                                                   self.H_,
                                                                                   self.w_,
                                                                                   y,
                                                                                   U=self.u_,
                                                                                   V=self.v_,
                                                                                   user_id=i)
                                                   for i, y in enumerate(bigrams))
        toc = time.time()
        L.debug('  [BIAS] Initialized %d subproblems in %.3f seconds',
                len(subproblems),
                toc - tic)

        # Collect usage statistics over subproblems
        counts = np.zeros(n_songs)
        duals = []
        ci = []

        b = self.b_.copy()

        tic = time.time()
        for sp in subproblems:
            ids_i = sp[-1]
            counts[ids_i] += 1.0
            duals.append(np.zeros(len(ids_i)))
            ci.append(np.take(b, ids_i))

        # Initialize ADMM parameters

        rho = 1.0

        for step in range(self.max_admm_iter):
            tic = time.time()
            ci = Parallel(n_jobs=self.n_jobs)(delayed(item_bias_optimize)(i,
                                                                          subproblems[i][0],
                                                                          subproblems[i][1],
                                                                          subproblems[i][2],
                                                                          duals[i],
                                                                          rho,
                                                                          self.u_,
                                                                          self.v_,
                                                                          b,
                                                                          Cout=ci[i])
                                              for i in range(len(subproblems)))
            toc = time.time()
            L.debug('  [BIAS] [%3d/%3d] Solved %d subproblems in %.3f seconds',
                    step,
                    self.max_admm_iter,
                    len(subproblems),
                    toc - tic)

            # Kill the old b
            tic = time.time()
            b.fill(0.0)
            for sp_i, c_i, d_i in zip(subproblems, ci, duals):
                ids_i = sp_i[-1]
                b[ids_i] += c_i + d_i

            # Compute the normalization factor
            my_norm = 1.0/(counts + self.bias_reg / rho)

            b[:] = my_norm * b
            toc = time.time()
            L.debug('  [BIAS] [%3d/%3d] Gathered solutions in %.3f seconds',
                    step,
                    self.max_admm_iter,
                    toc - tic)

            # Update the residuals
            tic = time.time()
            for sp_i, c_i, d_i in zip(subproblems, ci, duals):
                ids_i = sp_i[-1]
                d_i[:] = d_i + c_i - np.take(b, ids_i)
            toc = time.time()
            L.debug('  [BIAS] [%3d/%3d] Updated %d residuals in %.3f seconds',
                    step,
                    self.max_admm_iter,
                    len(subproblems),
                    toc - tic)

        # Save the results
        self.b_[:] = b

    def _fit_edges(self, bigrams):
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
                                                                    self.b_, y)
                                         for (idx, y) in enumerate(bigrams)):
            Z += Z_i
            num_usage += nu_i
            num_playlists += np_i

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
            Hexpw_norm[:] = [scipy.misc.logsumexp(np.take(w, hid.indices))
                             for hid in self.H_]

            obj += num_usage.dot(Hexpw_norm) + num_playlists * lse_w

            grad += np.ravel((num_usage * np.exp(-Hexpw_norm)).dot(Hexpw))
            grad += exp_w * (num_playlists * np.exp(-lse_w))

            return obj, grad

        w0 = self.w_.copy()

        bounds = [(-_EXP_BOUND, _EXP_BOUND)] * len(w0)
        w_opt, value, diag = scipy.optimize.fmin_l_bfgs_b(__edge_objective,
                                                          w0,
                                                          bounds=bounds)

        # Ensure that convergence happened correctly
        assert diag['warnflag'] == 0

        self.w_[:] = w_opt

    def fit(self, playlists, H):
        '''fit the model.

        :parameters:
          - playlists : dict (users => list of playlists)

            eg,
                playlists = {'bm106': [ [23, 35, 41, 32, 39],
                                        [18, 19, 72, 4],
                                        [12, 9] ] }

          - H : scipy.sparse.csr_matrix (n_songs, n_edges)
        '''

        # Stash the hypergraph and its transpose
        self.H_ = H.tocsr()
        self.H_T_ = H.T.tocsr()

        n_songs, n_edges = self.H_.shape

        # Convert playlists to bigrams
        self.user_map_, bigrams = make_bigrams_usermap(playlists)

        # Initialize edge weights
        if self.w_ is None:
            self.w_ = np.zeros(n_edges)

        if self.b_ is None:
            self.b_ = np.zeros(n_songs)

        if self.u_ is None:
            # Initialize to 0 by default
            self.u_ = np.zeros((len(playlists), self.n_factors))

        if self.v_ is None:
            # Initialize to random by default
            self.v_ = np.random.randn(n_songs, self.n_factors)

        # Training loop

        for iteration in range(self.max_iter):
            # Order of operations:

            if 'e' in self.params:
                L.info('[%3d/%3d] Fitting edge weights',
                       iteration, self.max_iter)
                self._fit_edges(bigrams)

            if 'b' in self.params:
                L.info('[%3d/%3d] Fitting song bias',
                       iteration, self.max_iter)
                self._fit_bias(bigrams)

            if 'u' in self.params:
                L.info('[%3d/%3d] Fitting user factors',
                       iteration, self.max_iter)
                self._fit_users(bigrams)

            if 's' in self.params:
                L.info('[%3d/%3d] Fitting song factors',
                       iteration, self.max_iter)
                self._fit_songs(bigrams)

        L.info('Done.')

    def sample(self, user_id=None, user_factor=None,
               n_songs=10, song_init=None, edge_init=None):
        '''Sample a playlist from the trained model.

        :parameters:

            - user_id : key into the usermap
            - user_factor : optional factor vector for an imaginary user
                - if neither are provided, the all zeros vector is used instead
            - n_songs : int > 0
                number of songs to sample
            - song_init : optional, int
                index of a pre-selected first song
            - edge_init : optional, int
                index of a pre-selected first edge

        :returns:
            - playlists : list
                list of track numbers
            - edges : list
                list of edge selections corresponding to selected tracks
        '''

        item_scores = np.zeros(self.H_.shape[0])

        if self.n_factors > 0:
            if user_factor is None:
                # The default is all zeros
                user_factor = np.zeros(self.n_factors)

                if user_id in self.user_map_:
                    user_factor = self.u_[self.user_map_[user_id]]
                else:
                    raise ValueError('Unknown user_id: {0}'.format(user_id))

            # Score the items
            item_scores = self.v_.dot(user_factor)

        if self.b_ is not None:
            item_scores += self.b_

        item_scores = np.exp(item_scores)

        expw = np.exp(self.w_)

        if edge_init is not None:
            edge = edge_init
        else:
            if song_init is not None:
                # Draw the initial edge from the song-conditional distribution
                edge = categorical(self.H_[song_init].multiply(expw))
            else:
                # Draw the initial edge from the song-conditional distribution
                edge = categorical(expw)

        playlist = []
        edges = []

        for i in range(n_songs):
            # Pick a song from the current edge
            song = categorical(self.H_T_[edge].multiply(item_scores))

            playlist.append(song)
            edges.append(edge)

            # Pick an edge from the current song
            edge = categorical(self.H_[song].multiply(expw))

        return playlist, edges

    def loglikelihood(self, playlist, user_id=None, user_num=None,
                      normalize=False):
        '''Compute the log-likelihood of a single playlist.

        :parameters:
            playlist : list of ints
              The playlist

            user_id : hashable or None
              If supplied, the key for the user

            user_num : int or None
              If supplied, the index number of the user
              If no user data is supplied, we synthesize an
              all-zeros user factor

            normalize : bool
              If true, loglikelihood will be normalized by length

        :returns:
            - loglikelihood : float
              The log probability of generating the observed sequence
        '''

        user_factor = np.zeros(self.n_factors)

        if user_id is not None:
            user_num = self.user_map_[user_id]

        if user_num is not None:
            user_factor = self.u_[user_num]

        # Score the items
        item_scores = np.zeros(self.H_.shape[0])

        if self.v_ is not None:
            item_scores += self.v_.dot(user_factor)

        if self.b_ is not None:
            item_scores += self.b_

        item_scores = np.exp(item_scores)

        # Compute edge probabilities
        edge_scores = np.exp(self.w_)

        # Build bigrams
        bigrams = playlist_to_bigram(playlist)

        # Compute likelihoods
        ll = 0.0
        for s, t in bigrams:
            # Compute the edge distribution P(e | s)
            if s is None:
                # All edges are valid
                p_e_s = edge_scores
            else:
                # Only edges touching s are valid
                p_e_s = self.H_[s].multiply(edge_scores)

            # Normalize to form a distribution
            p_e_s = np.ravel(p_e_s) / np.sum(p_e_s)

            # Compute P(t | e) = score(t) / sum(score(j) | j in E)
            edge_mass = self.H_T_ * item_scores
            p_t_e = item_scores[t] / edge_mass

            # Compute P(t | s) = P(t | e) P(e | s)
            ll += np.log(p_t_e.dot(p_e_s))

        if normalize:
            ll /= len(playlist)

        return ll


# Static methods: things that can parallelize
def make_bigrams_usermap(playlists):
    '''generate user map and bigram lists.

    input:
        - playlists : dict : user_id => playlists

    '''

    user_map = dict()

    bigrams = []

    for i, user_id in enumerate(sorted(playlists)):

        user_map[user_id] = i

        user_bigrams = []
        for pl in playlists[user_id]:
            user_bigrams.extend(playlist_to_bigram(pl))

        bigrams.append(user_bigrams)

    return user_map, bigrams


def playlist_to_bigram(playlist):
    '''Convert a sequence of ids into bigram form.

    A 'None' is pushed onto the front to indicate the beginning.
    '''

    my_pl = [None]
    my_pl.extend(playlist)

    bigrams = zip(my_pl[:-1], my_pl[1:])

    return bigrams


def categorical(z):
    '''Sample from a categorical random variable'''

    z = np.ravel(np.asarray(z) / z.sum())

    assert np.all(z >= 0.0) and np.any(z > 0)

    return np.flatnonzero(np.random.multinomial(1, z))[0].astype(np.uint)


# common functions to user, item, and bias optimization:
def make_bigram_weights(H, s, t, weight):
    if s is None:
        # This is a phantom state, so we only care about t
        my_weight = H[t].multiply(weight)
    else:
        # Otherwise, (s,t) is a valid transition, so use both
        my_weight = H[s].multiply(H[t]).multiply(weight)

    my_weight = np.ravel(my_weight)

    # Normalize the edge probabilities
    return my_weight / my_weight.sum()


def sample_noise_items(n_neg, H, edge_dist, b, y_pos):
    '''Sample n_neg items from the noise distribution,
    forbidding observed samples y_pos.
    '''

    y_forbidden = set(y_pos)

    edge_dist = np.asarray(edge_dist / edge_dist.sum())

    # Our item distribution will be softmax over bias, ignoring the user factor
    full_item_dist = np.exp(b)

    # Knock out forbidden items
    full_item_dist[list(y_forbidden)] = 0.0

    noise_ids = []

    while len(noise_ids) < n_neg:
        # Sample an edge
        edge_id = categorical(edge_dist)

        item_dist = np.ravel(H[:, edge_id].T.multiply(full_item_dist))

        item_dist_norm = np.sum(item_dist)

        item_dist /= item_dist_norm

        if not np.all(np.isfinite(item_dist)):
            break

        while True:
            new_item = categorical(item_dist)

            if new_item not in y_forbidden:
                break

        y_forbidden.add(new_item)
        noise_ids.append(new_item)
        full_item_dist[new_item] = 0.0

    return noise_ids


def generate_user_instance(n_neg, H, edge_dist, bigrams, b=None,
                           U=None, V=None, user_id=None):
    '''Generate a subproblem instance.

    By default, negatives will be sampled according to their bias term `b`.

    If latent factors and a user id are supplied, negatives will be sampled
    according to their unbiased, personalized scores `U[user_id].dot(V)`

    Inputs:
        - n_neg : # of negative samples
        - H : hypergraph incidence matrix
        - edge_dist : weights of the edges of H
        - b : bias factors for items
        - bigrams : list of tuples (s, t) for the user
        - U : user factors
        - V : item factors
        - user_id : index of the user

    Outputs:
        - y : +-1 label vector
        - weights : importance weights, shape=y.shape
        - ids : list of indices for the sampled points, shape=y.shape
    '''

    if b is None:
        if user_id is not None:
            item_scores = V.dot(U[user_id])
        else:
            item_scores = np.ones(H.shape[0])
    else:
        item_scores = b

    # 1. Extract positive ids
    pos_ids = [t for (s, t) in bigrams]

    exp_w = np.exp(edge_dist)

    # 2. Sample n_neg songs from the noise model (u=0)
    noise_ids = sample_noise_items(n_neg,
                                   H,
                                   exp_w,
                                   item_scores,
                                   pos_ids)

    # 3. Compute and normalize the bigram transition weights
    #   handle the special case of s==None here

    bigram_weights = np.asarray([make_bigram_weights(H, s, t, exp_w)
                                 for (s, t) in bigrams])

    # 4. Compute the importance weights for noise samples
    noise_weights = np.sum(np.asarray([[(H[i] * bg.T)
                                        for i in noise_ids]
                                       for bg in bigram_weights]),
                           axis=0).ravel()

    # 5. Construct the inputs to the solver
    y = np.ones(len(pos_ids) + len(noise_ids))
    y[len(pos_ids):] = -1

    # The first bunch are positive examples, and get weight=+1
    weights = np.ones_like(y)

    # The remaining examples get noise weights
    weights[len(pos_ids):] = noise_weights

    ids = np.concatenate([pos_ids, noise_ids]).astype(np.int)

    return y, weights, ids


# edge optimization
def edge_user_weights(H, H_T, u, idx, v, b, bigrams):
    '''Compute the edge weights and transition statistics for a user.'''

    # First, compute the user-item affinities
    item_scores = np.zeros(H.shape[0])
    if u is not None and v is not None:
        item_scores += v.dot(u[idx])
    if b is not None:
        item_scores += b

    item_scores = np.exp(item_scores)

    # Now aggregate by edge
    edge_scores = (H_T * item_scores)**(-1.0)

    # num playlists is the number of bigrams where s == None
    num_playlists = 0
    num_usage = np.zeros(len(b))
    Z = 0

    # Now sum over bigrams
    for s, t in bigrams:
        Z = Z + make_bigram_weights(H, s, t, edge_scores)
        if s is None:
            num_playlists += 1
        else:
            num_usage[s] += 1

    return Z, num_usage, num_playlists


# item optimization
def item_factor_optimize(i, y, weights, ids, dual, rho, U_, V_, b_,
                         Aout=None):

    # Slice out the relevant components of this subproblem
    u = U_[i]
    V = np.take(V_, ids, axis=0)
    b = np.take(b_, ids)

    # Compute the residual
    Z = V - dual

    def __item_obj(_a):
        '''Optimize the item objective function'''

        A = _a.view()
        A.shape = Z.shape

        scores = y * (A.dot(u) + b)

        delta = A - Z
        f = rho * 0.5 * np.sum(delta**2)
        f += weights.dot(np.logaddexp(0, -scores))

        grad = rho * delta
        grad += np.multiply.outer(-y * weights / (1.0 + np.exp(scores)), u)

        grad_out = grad.view()
        grad_out.shape = (grad.size, )

        return f, grad_out

    # Probably a decent initial point
    if Aout is not None:
        a0 = Aout
    else:
        # otherwise, V slice is pretty good too
        a0 = V.copy()

    a_opt, value, diagnostic = scipy.optimize.fmin_l_bfgs_b(__item_obj, a0)

    # Ensure that convergence happened correctly
    assert diagnostic['warnflag'] == 0

    # Reshape the solution
    a_opt.shape = V.shape

    # If we have a target destination, fill it
    if Aout is not None:
        Aout[:] = a_opt
    else:
        # Otherwise, point to the new array
        Aout = a_opt

    return Aout


# bias optimization
def item_bias_optimize(i, y, weights, ids, dual, rho, U_, V_, b_, Cout=None):

    # Slice out the relevant components of this subproblem
    b = np.take(b_, ids)

    # Compute the residual
    z = b - dual

    if U_ is not None:
        u = U_[i]
        V = np.take(V_, ids, axis=0)
        user_scores = V.dot(u)
    else:
        user_scores = np.zeros_like(b)

    def __bias_obj(c):

        scores = y * (user_scores + c)

        delta = c - z

        f = rho * 0.5 * np.sum(delta**2)

        f += weights.dot(np.logaddexp(0, -scores))

        grad = rho * delta
        grad += -y * weights / (1.0 + np.exp(scores))

        return f, grad

    if Cout is not None:
        c0 = Cout
    else:
        c0 = b.copy()

    c_opt, value, diagnostic = scipy.optimize.fmin_l_bfgs_b(__bias_obj, c0)

    assert diagnostic['warnflag'] == 0

    if Cout is not None:
        Cout[:] = c_opt
    else:
        Cout = c_opt

    return Cout


# user optimization
def user_problem(n_noise, H, w, reg, v, b, bigrams, u0=None):
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

    :returns:
        - u_opt : ndarray, shape=(n_factors,)
          Optimal user vector
    '''

    y, weights, ids = generate_user_instance(n_noise, H, w, bigrams, b=b)

    return user_optimize_objective(reg,
                                   np.take(v, ids, axis=0),
                                   np.take(b, ids),
                                   y,
                                   weights,
                                   u0=u0)


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

            min_u reg * ||u||^2
                + sum_i y[i] * omega[i] * log(1 + exp(-y * u'v[i] + b[i]))

        '''

        # Compute the scores
        scores = y * (v.dot(u) + b)

        f = reg * 0.5 * np.sum(u**2)
        f += omega.dot(np.logaddexp(0, -scores))

        grad = reg * u
        grad -= v.T.dot(y * omega / (1.0 + np.exp(scores)))

        return f, grad

    if not u0:
        u0 = np.zeros(v.shape[-1])
    else:
        assert len(u0) == v.shape[-1]

    u_opt, value, diagnostic = scipy.optimize.fmin_l_bfgs_b(__user_obj, u0)

    # Ensure that convergence happened correctly
    assert diagnostic['warnflag'] == 0

    return u_opt
