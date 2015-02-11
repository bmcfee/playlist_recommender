#!/usr/bin/env python
"""stochastic hypergraph recommended playlists"""

import numpy as np

import logging

import theano
import theano.tensor as T
import theano.sparse as ts

import lasagne

from sklearn.base import BaseEstimator

# Prevent numerical underflow
_EPS = 1e-8

L = logging.getLogger(__name__)


class PlaylistModel(BaseEstimator):
    '''Personalized hypergraph random walk playlist model'''

    def __init__(self, H, n_users,
                 n_factors=4, edge_reg=1e-3, bias_reg=1e-3, user_reg=1e-3,
                 song_reg=1e-3, n_epochs=10, batch_size=512,
                 edge_init=None, bias_init=None,
                 user_init=None, song_init=None,
                 params='ebus', verbose=0,
                 dropout=0.0,
                 callback=None):
        """Initialize a personalized playlist model

        :parameters:
         - H : scipy.sparse matrix [shape=(n_songs, n_edges)]
            Hypergraph edge-incidence matrix

         - n_users : int > 0
            Number of users in the model

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

         - n_epochs: int > 0
            Number of optimization steps

         - batch_size: int > 0
            Number of examples to use in each training batch

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

         - dropout : float in [0, 1.0)
            If > 0, alternative items are randomly dropped during training

         - verbose : int >= 0
            Verbosity (logging) level

         - callback : None or callable
            An optional function to call after each iteration
            This can be used for validation or versioning.

            Signature:
            callback(model_object)
        """

        # If we don't learn latent factors,
        # set n_factors to 1 and pin the user variables to 0
        if 'u' not in params and 's' not in params:
            n_factors = 1

        # Stash the hypergraph as CSR
        self.H = H.tocsr().astype(theano.config.floatX)

        self.n_users = n_users

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.params = params
        self.n_factors = n_factors

        self.dropout = dropout

        if user_init is not None:
            self.n_factors = user_init.shape[1]

        if song_init is not None:
            self.n_factors = song_init.shape[1]

        self.edge_reg = edge_reg
        self.bias_reg = bias_reg
        self.user_reg = user_reg
        self.song_reg = song_reg

        self.verbose = verbose
        self.callback = callback

        L.setLevel(self.verbose)

        self.init_variables(edge_init,
                            bias_init,
                            user_init,
                            song_init)

        self.init_functions()

    def init_variables(self, edge_init, bias_init, user_init, song_init):
        '''Construct theano shared variables'''

        self.user_map_ = {}

        self.n_songs, self.n_edges = self.H.shape
        dtype = theano.config.floatX

        # Initialize the edge weights
        if edge_init is None:
            edge_init = np.zeros(self.n_edges, dtype=dtype)
        else:
            assert len(edge_init) == self.n_edges
            edge_init = edge_init.astype(dtype)

        self._w = theano.shared(edge_init, name='w')

        # Initialize the bias term
        if bias_init is None:
            bias_init = np.zeros(self.n_songs, dtype=dtype)
        else:
            assert len(bias_init) == self.n_songs
            bias_init = bias_init.astype(dtype)

        self._b = theano.shared(bias_init, name='b')

        # Initialize the user factors
        if user_init is None:
            user_init = np.zeros((self.n_users, self.n_factors), dtype=dtype)
        else:
            assert np.allclose(user_init.shape, (self.n_users, self.n_factors))
            user_init = user_init.astype(dtype)

        self._U = theano.shared(user_init, name='U')

        # Initialize the song factors
        if song_init is None:
            song_init = np.random.randn(self.n_songs,
                                        self.n_factors).astype(dtype)
        else:
            assert np.allclose(song_init.shape, (self.n_songs, self.n_factors))
            song_init = song_init.astype(dtype)

        self._V = theano.shared(song_init, name='V')

        self._rng = theano.sandbox.rng_mrg.MRG_RandomStreams()

    def fit(self, playlists):
        '''fit the model.

        :parameters:
          - playlists : dict (users => list of playlists)

            eg,
                playlists = {'bm106': [ [23, 35, 41, 32, 39],
                                        [18, 19, 72, 4],
                                        [12, 9] ] }
        '''

        # Decompose playlists into (user, source, target) tuples
        self.init_user_map(playlists)

        u_i, y_s, y_t = make_theano_inputs(playlists, self.user_map_)

        # Training loop
        self.nll_ = []
        self.cost_ = []

        for epoch in range(self.n_epochs):

            self.epochs_ = epoch
            # Generate a random permutation
            idx = np.random.permutation(np.arange(len(u_i)))

            L.debug('Training epoch {:d}'.format(self.epochs_))

            for i in range(0, len(idx), self.batch_size):

                b_ll, b_cost = self._train(u_i=u_i[idx[i:i+self.batch_size]],
                                           y_s=y_s[idx[i:i+self.batch_size]],
                                           y_t=y_t[idx[i:i+self.batch_size]],
                                           p=self.dropout)
                self.nll_.append(b_ll)
                self.cost_.append(b_cost)

                if hasattr(self.callback, '__call__'):
                    self.callback(self)

        self.nll_ = np.asarray(self.nll_)
        self.cost_ = np.asarray(self.cost_)

        L.info('Done.')

    def init_functions(self):
        '''Construct functions for the model'''

        # Construct the objective function

        #   Input variables
        u_i, y_s, y_t = T.ivectors(['u_i', 'y_s', 'y_t'])

        dropout = T.fscalar(name='p')

        #   Intermediate variables: n_examples * n_songs
        item_scores = T.dot(self._U[u_i], self._V.T) + self._b

        # subtract off the row-wise max for numerical stability
        item_scores = item_scores - item_scores.max(axis=1, keepdims=True)

        e_scores = T.exp(item_scores)

        if T.gt(dropout, 0.0):
            # Construct a random dropout mask
            retain_prob = 1.0 - dropout
            M = self._rng.binomial(e_scores.shape,
                                   p=retain_prob,
                                   dtype=theano.config.floatX)

            # Importance weight so that E[M[i,j]] = 1
            M /= retain_prob

            # The positive examples should always be sampled
            M = theano.tensor.set_subtensor(M[T.arange(y_t.shape[0]), y_t],
                                            1.0)

            e_scores = e_scores * M

        #   Edge feasibilities: n_examples * n_edges
        prev_feas = sparse_slice_rows(self.H, y_s)
        #   Detect and reset initial-state transitions
        prev_feas = theano.tensor.set_subtensor(prev_feas[y_s < 0, :], 1)

        #   Raw edge probabilities: n_examples * n_edges
        edge_given_prev = T.nnet.softmax(prev_feas * self._w)

        #   Compute edge normalization factors: n_examples * n_edges
        #     sum of score mass in each edge for each user
        edge_norms = ts.dot(e_scores, self.H)

        #   Slice the edge weights according to incoming feasibilities: n_examples
        next_weight = e_scores[T.arange(y_t.shape[0]), y_t]

        #   Marginalize: n_examples * n_edges
        next_feas = sparse_slice_rows(self.H, y_t)

        probs = next_weight * T.sum(next_feas * (edge_given_prev / (_EPS + edge_norms)),
                                    axis=1,
                                    keepdims=True)

        # Data likelihood term
        ll = T.log(probs)
        avg_ll = ll.mean()

        # Priors
        w_prior = -0.5 * self.edge_reg * (self._w**2).sum()
        b_prior = -0.5 * self.bias_reg * (self._b**2).sum()
        u_prior = -0.5 * self.user_reg * (self._U**2).sum()
        v_prior = -0.5 * self.song_reg * (self._V**2).sum()

        # negative log-MAP objective
        cost = -1.0 * (avg_ll + u_prior + v_prior + b_prior + w_prior)

        # Construct the updates
        variables = []
        if 'e' in self.params:
            variables.append(self._w)
        if 'b' in self.params:
            variables.append(self._b)
        if 'u' in self.params:
            variables.append(self._U)
        if 's' in self.params:
            variables.append(self._V)

        updates = lasagne.updates.adagrad(cost, variables)

        self._train = theano.function(inputs=[u_i, y_s, y_t, dropout],
                                      outputs=[avg_ll, cost],
                                      updates=updates)

        self._loglikelihood = theano.function(inputs=[u_i, y_s, y_t,
                                                      theano.Param(dropout,
                                                                   default=0.0,
                                                                   name='p')],
                                              outputs=[ll])

    @property
    def U_(self):
        assert hasattr(self, '_U')
        return self._U.get_value()

    @property
    def V_(self):
        assert hasattr(self, '_V')
        return self._V.get_value()

    @property
    def b_(self):
        assert hasattr(self, '_b')
        return self._b.get_value()

    @property
    def w_(self):
        assert hasattr(self, '_w')
        return self._w.get_value()

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

        item_scores = np.zeros(self.H.shape[0])

        if self.n_factors > 0:
            if user_factor is None:
                # The default is all zeros
                user_factor = np.zeros(self.n_factors)

                if user_id in self.user_map_:
                    user_factor = self.U_[self.user_map_[user_id]]
                else:
                    raise ValueError('Unknown user_id: {0}'.format(user_id))

            # Score the items
            item_scores = self.V_.dot(user_factor)

        if self.b_ is not None:
            item_scores += self.b_

        item_scores = np.exp(item_scores)

        expw = np.exp(self.w_)

        if edge_init is not None:
            edge = edge_init
        else:
            if song_init is not None:
                # Draw the initial edge from the song-conditional distribution
                edge = categorical(self.H[song_init].multiply(expw))
            else:
                # Draw the initial edge from the song-conditional distribution
                edge = categorical(expw)

        playlist = []
        edges = []

        for _ in range(n_songs):
            # Pick a song from the current edge
            song = categorical(self.H.T[edge].multiply(item_scores))

            playlist.append(song)
            edges.append(edge)

            # Pick an edge from the current song
            edge = categorical(self.H[song].multiply(expw))

        return playlist, edges

    def loglikelihood(self, playlists, avg=True):
        '''Compute the average log-likelihood of a collection of playlists'''

        u_i, y_s, y_t = make_theano_inputs(playlists, user_map=self.user_map_)

        # Compute in batches
        n_examples = len(u_i)

        ll = []

        for i in range(0, n_examples, self.batch_size):
            bll = self._loglikelihood(u_i=u_i[i:i+self.batch_size],
                                      y_s=y_s[i:i+self.batch_size],
                                      y_t=y_t[i:i+self.batch_size])[0].ravel()
            ll.extend(list(bll))

        ll = np.asarray(ll)

        if avg:
            ll = ll.mean()

        return ll

    def init_user_map(self, playlists):
        '''Build a mapping of user ids from a dictionary of playlists'''

        self.user_map_ = dict()

        for i, user_id in enumerate(playlists.iterkeys()):
            self.user_map_[user_id] = i

    def serialize(self):
        '''Serialize the relevant parameters of the model'''
        return dict(params=self.get_params(),
                    w=self.w_,
                    b=self.b_,
                    U=self.U_,
                    V=self.V_,
                    usermap=self.user_map_,
                    H=self.H)


# Static functions
def make_theano_inputs(playlists, user_map):
    '''Given a dictionary on user -> list of playlists,
    and a dictionary of user -> user_id,
    Construct theano-friendly inputs.
    '''

    u_id = []
    y_s = []
    y_t = []

    for user_key, pls in playlists.iteritems():

        my_uid = user_map[user_key]

        for pl in pls:
            prevs, nexts = playlist_to_bigrams(pl)

            u_id.extend([my_uid] * len(prevs))
            y_s.extend(prevs)
            y_t.extend(nexts)

    return (np.asarray(u_id, dtype=np.int32),
            np.asarray(y_s, dtype=np.int32),
            np.asarray(y_t, dtype=np.int32))


def playlist_to_bigrams(playlist, default=-1):
    '''Convert a sequence of ids into bigram form.

    A 'None' is pushed onto the front to indicate the beginning.
    '''

    my_pl = [default]
    my_pl.extend(playlist)

    return my_pl[:-1], my_pl[1:]


def to_one_hot(y, nb_class, dtype=None):
    """Return a matrix where each row correspond to the one hot
    encoding of each element in y.

        :param y: A vector of integer value between 0 and nb_class - 1.
        :param nb_class: The number of class in y.
        :param dtype: The dtype of the returned matrix. Default floatX.

        :return: A matrix of shape (y.shape[0], nb_class), where each
          row ``i`` is the one hot encoding of the corresponding ``y[i]``
          value.
    """

    ret = theano.tensor.zeros((y.shape[0], nb_class),
                              dtype=dtype)

    ret = theano.tensor.set_subtensor(ret[theano.tensor.arange(y.shape[0]), y],
                                      1)
    return ret


def sparse_slice_rows(H, idx):
    '''Returns a dense slice H[idx, :]'''

    vecs = to_one_hot(idx, H.shape[0], dtype=H.dtype)

    return ts.dot(vecs, H)


def categorical(z, size=None, replace=True):
    '''Sample from a categorical random variable'''

    z = np.ravel(z).astype(np.float64)
    z /= np.sum(z)

    assert np.all(z >= 0.0) and np.any(z > 0)

    return np.random.choice(len(z), size=size, p=z, replace=replace)
