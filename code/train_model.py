#!/usr/bin/env python
# CREATED:2015-02-11 12:31:12 by Brian McFee <brian.mcfee@nyu.edu>


import argparse
import sys
import shyrp
import numpy as np
import scipy.sparse
import cPickle as pickle

import pandas as pd

np.set_printoptions(precision=3)

shyrp.theano.config.exception_verbosity = 'high'
shyrp.theano.config.floatX = 'float32'

MIN_EDGE_SIZE = 318
EDGE_REG = 1e-6
BIAS_REG = 1e-6
NUM_EPOCHS = 10
BATCH_SIZE = 128
VERBOSE = shyrp.logging.DEBUG


def decompose(df, song_map, max_users=np.inf):
    '''Crunch a playlist dataframe into shyrp-friendly format'''

    playlists = {}

    if max_users == -1:
        max_users = np.inf

    for count, user in enumerate(df.index.levels[0].unique()):
        if count >= max_users:
            break
        # Add the user to the playlist collection
        playlists.setdefault(user, [])
        # Slice the frame
        d_u = df.loc[user]
        for mix_id in d_u.index.get_level_values(0).unique():
            d_mix = d_u.loc[mix_id]
            for segment_id in d_mix.index.get_level_values(0).unique():
                playlists[user].append([song_map[_]
                                        for _ in d_mix.loc[segment_id]['song_id']])
    return playlists


def graph_to_song_map(H):
    '''pull the song id to row number index'''   
    return dict([_[::-1] for _ in enumerate(H.index)])


def load_edges(*files, **kwargs):

    min_size = kwargs.get('min_size', MIN_EDGE_SIZE)

    frames = [pd.read_pickle(_) for _ in files]
    frames = [frame.drop(frame.columns[frame.count() < min_size],
                         axis=1).to_dense()
              for frame in frames]

    H = frames.pop(0)

    while len(frames):
        H = H.join(frames.pop(0), how='outer', sort=True)

    H = H.to_sparse()
    return H


def run_experiment(edge=False, bias=False, user=False, song=False,
                   max_users=-1, playlists='', edges=None,
                   output='', num_factors=0):

    params = ''
    if edge:
        params += 'e'
    if bias:
        params += 'b'
    if user:
        params += 'u'
    if song:
        params += 's'

    if params == '':
        raise RuntimeError('At least one model parameter must be set: {EBUS}.')

    # Load the graph
    H_frame = load_edges(*edges)
    H_frame = H_frame.to_sparse(fill_value=0.0)

    # And make a sparse matrix for shyrp
    H = scipy.sparse.csr_matrix(H_frame.values, dtype=np.float32)

    # Pull out the song ids
    songs = graph_to_song_map(H_frame)
    song_ids = dict([_[::-1] for _ in songs.items()])

    # Load the training data
    pl_train = pd.read_pickle(playlists)

    playlists = decompose(pl_train, songs, max_users=max_users)

    model = shyrp.PlaylistModel(H, len(playlists),
                                edge_reg=EDGE_REG,
                                bias_reg=BIAS_REG,
                                n_factors=num_factors,
                                n_epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                params='eb',
                                verbose=VERBOSE)

    model.fit(playlists)

    with open(output, 'w') as fdesc:
        pickle.dump({'model': model,
                     'train_score': model.loglikelihood(playlists),
                     'baseline': -np.log(H.shape[0]),
                     'song_ids': song_ids},
                    fdesc, protocol=-1)


def process_arguments(args):

    parser = argparse.ArgumentParser(description='SHYRP driver')

    parser.add_argument('-e', '--edge', dest='edge', default=False,
                        action='store_true', help='Learn edge weights')
    parser.add_argument('-b', '--bias', dest='bias', default=False,
                        action='store_true', help='Learn song bias')
    parser.add_argument('-u', '--user', dest='user', default=False,
                        action='store_true', help='Learn user factors')
    parser.add_argument('-s', '--song', dest='song', default=False,
                        action='store_true', help='Learn song factors')
    parser.add_argument('-o', '--output', dest='output', required=True,
                        type=str, help='Output path for trained model')
    parser.add_argument('-m', '--max-users', dest='max_users', type=int,
                        default=-1, help='Maximum number of users to train on')
    parser.add_argument('-d', '--num-factors', dest='num_factors', type=int,
                        help='Number of latent factors')
    parser.add_argument('playlists', required=True, type=str,
                        help='Playlist data pickle')
    parser.add_argument('edges', nargs='+',
                        help='One or more edge files (dataframe pickles)')

    return vars(parser.parse_args(args))

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])

    run_experiment(**params)
